import argparse
import safety_gymnasium as gym
import os
import sys
import pickle

# Call utilities
from utils import *

# Call models
from models.continuous_policy import Policy
from models.critic import Value
from models.discrete_policy import DiscretePolicy

# Call algorithms
from algos.CPO import CPO
from algos.CPOMeta import CPOMeta

# Call tensorboard for logging
from torch.utils.tensorboard import SummaryWriter

# Returns the current local date
from datetime import date


def main_loop():
    today = date.today()
    print("Today date is: ", today)

    # Parse arguments
    args = parse_all_arguments()
    print("Arguments: ", args)

    """Data type and compute device"""
    args.dtype = torch.float64
    torch.set_default_dtype(args.dtype)

    args.device = select_device()

    """environment"""
    envs = create_envs(args)

    state_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.shape[0]

    print(f"\nstate-action dim: {state_dim}/{action_dim}\n")

    is_disc_action = len(envs[0].action_space.shape) == 0
    running_state = ZFilter((state_dim,), clip=5)

    """seeding"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    """create all the paths to save learned models/data"""
    save_info_obj = save_info(
        assets_dir(), args.exp_num, args.exp_name, args.env_name
    )  # model saving object
    save_info_obj.create_all_paths()  # create all paths
    writer = SummaryWriter(
        os.path.join(assets_dir(), save_info_obj.saving_path, "runs/")
    )  # tensorboard summary

    """define actor and critic"""
    if args.model_path is None:
        if is_disc_action:
            policy_net = DiscretePolicy(state_dim, action_dim)
        else:
            policy_net = Policy(state_dim, action_dim, log_std=args.log_std)
        value_net = Value(state_dim)
        cost_net = Value(state_dim)
    else:
        print("TRAINING FROM PREVIOUS PARAMETERS. . .", args)
        policy_net, value_net, cost_net, running_state, prev_args = pickle.load(
            open(args.model_path + "model.p", "rb")
        )
        prev_args.model_path = args.model_path
        prev_args.update_iter_num = args.update_iter_num
        prev_args.max_iter_num = args.max_iter_num
        prev_args.meta_iter_num = args.meta_iter_num
        prev_args.is_meta_test = args.is_meta_test
        args = prev_args

    policy_net.to(args.device)
    value_net.to(args.device)
    cost_net.to(args.device)
    if args.is_meta_test:
        print("meta testing")
        algo = CPO(
            envs,
            policy_net,
            value_net,
            cost_net,
            args,
            running_state=running_state,
            num_threads=args.num_threads,
        )
        algo.train_CPO(writer, save_info_obj)
    else:
        """create agent"""
        if args.algo_name == "CPO":
            algo = CPO(
                envs,
                policy_net,
                value_net,
                cost_net,
                args,
                args.dtype,
                args.device,
                running_state=running_state,
            )
            algo.train_CPO(writer, save_info_obj)

        elif args.algo_name == "CPOMeta":
            algo = CPOMeta(
                envs,
                policy_net,
                value_net,
                cost_net,
                writer,
                args,
                running_state=running_state,
            )
            algo.train_CPOMeta(save_info_obj)

main_loop()
