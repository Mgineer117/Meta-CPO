import argparse
#import gym
import safety_gymnasium as gym
#from gym import utils
import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
import statistics as st
from models.continuous_policy import Policy
from models.critic import Value
from models.discrete_policy import DiscretePolicy
from algos.TRPO import TRPO
from algos.CPO import CPO
from algos.TRPOMeta import TRPOMeta
from algos.CPOMeta import CPOMeta
from core.common import estimate_advantages, estimate_constraint_value

import pdb
CUDA_LAUNCH_BLOCKING=1

#summarizing using tensorboard
from torch.utils.tensorboard import SummaryWriter

# Returns the current local date
from datetime import date

def main_loop():
    today = date.today()
    print("Today date is: ", today)

    # Parse arguments 
    args = parse_all_arguments()
    print("Arguments: ",args)

    """Data type and compute device"""
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        print('using gpu')
        torch.cuda.set_device(args.gpu_index)

    """environment"""
    print()
    envs = []
    print(f'{1} Environement generated... with parameters:')
    env = gym.make(args.env_name + str(0) + '-v0')
    print('state dim: ',env.observation_space.shape[0])
    envs.append(env)
    for i in range(1, args.env_num+1):
        if i != args.env_num:
            print(f'{i+1} Environement generated... with parameters:')
            env = gym.make(args.env_name + str(1) + '-v0')
        else:
            print('Test Environement generated... with parameters:')
            env = gym.make(args.env_name + str(2) + '-v0')
        print('state dim: ',env.observation_space.shape[0])
        envs.append(env)
    print()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print('action dim: ', action_dim)

    is_disc_action = len(env.action_space.shape) == 0
    running_state = ZFilter((state_dim,), clip=5)

    """seeding"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    """create all the paths to save learned models/data"""
    save_info_obj = save_info(assets_dir(), args.exp_num, args.exp_name, args.env_name) #model saving object
    save_info_obj.create_all_paths() # create all paths
    writer = SummaryWriter(os.path.join(assets_dir(), save_info_obj.saving_path, 'runs/')) #tensorboard summary
    
    """define actor and critic"""
    if args.model_path is None:
        if is_disc_action:
            policy_net = DiscretePolicy(state_dim, env.action_space.n)
        else:
            policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std)
        value_net = Value(state_dim)
        cost_net = Value(state_dim)
    else:
        print('TRAINING FROM PREVIOUS PARAMETERS. . .', args)
        policy_net, value_net, cost_net, running_state, prev_args = pickle.load(open(args.model_path + 'model.p', "rb"))
        prev_args.model_path = args.model_path
        prev_args.update_iter_num = args.update_iter_num
        prev_args.max_iter_num = args.max_iter_num
        prev_args.meta_iter_num = args.meta_iter_num
        prev_args.is_meta_test = args.is_meta_test
        args = prev_args

    policy_net.to(device)
    value_net.to(device)
    cost_net.to(device)
    if args.is_meta_test:
        print('meta testing')
        algo = CPO(envs, policy_net, value_net, cost_net, args, dtype, device,
                    running_state=running_state, num_threads=args.num_threads)
        algo.train_CPO(writer, save_info_obj)
    else:
        """create agent"""
        if args.algo_name == 'TRPO':
            algo = TRPO(envs, policy_net, value_net, cost_net, args, dtype, device, 
                running_state=running_state, num_threads=args.num_threads)
            algo.train_TRPO(writer, save_info_obj)

        elif args.algo_name == 'TRPOMeta':
            algo = TRPOMeta(envs, policy_net, value_net, cost_net, args, dtype, device, 
                    running_state=running_state, num_threads=args.num_threads)
            algo.train_TRPOMeta(writer, save_info_obj)

        # CPO's optimization problem is solver-error prone.
        elif args.algo_name == 'CPO':
            algo = CPO(envs, policy_net, value_net, cost_net, args, dtype, device,
                        running_state=running_state, num_threads=args.num_threads)
            algo.train_CPO(writer, save_info_obj)

        elif args.algo_name == 'CPOMeta':
            algo = CPOMeta(envs, policy_net, value_net, cost_net, args, dtype, device,
                            running_state=running_state, num_threads=args.num_threads)
            algo.train_CPOMeta(writer, save_info_obj)


main_loop()


