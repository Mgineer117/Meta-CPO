import argparse
import gym
import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from itertools import count
from utils import *
import pdb

parser = argparse.ArgumentParser(description='Save expert trajectory')
parser.add_argument('--env-name', default="Ant-v4", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', metavar='G',
                    help='name of the expert model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--max-expert-state-num', type=int, default=50000, metavar='N',
                    help='maximal number of main iterations (default: 50000)')
parser.add_argument('--mean-action', action='store_true', default=True,
                    help='Sample mean action if True, o.w. sample a stochastic action')
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
env = gym.make(args.env_name)
env.seed(args.seed)
torch.manual_seed(args.seed)
is_disc_action = len(env.action_space.shape) == 0
state_dim = env.observation_space.shape[0]

policy_net, _, running_state = pickle.load(open(args.model_path, "rb"))
running_state.fix = True
expert_traj = []
reward_list = []

def main_loop():

    num_steps = 0

    for i_episode in count():

        state = env.reset()
        state = running_state(state)
        reward_episode = 0

        for t in range(10000):
            state_var = tensor(state).unsqueeze(0).to(dtype)
            
            if args.mean_action == True:
                # choose mean action
                action = policy_net(state_var)[0][0].detach().numpy()
            else:
                # choose stochastic action
                action = policy_net.select_action(state_var)[0].cpu().numpy()
            
            action = int(action) if is_disc_action else action.astype(np.float64)
            next_state, reward, done, _ = env.step(action)
            next_state = running_state(next_state)
            reward_episode += reward
            num_steps += 1

            expert_traj.append(np.hstack([state, action]))

            if args.render:
                env.render()
            if done or num_steps >= args.max_expert_state_num:
                break

            state = next_state

        print('Episode {}\t reward: {:.2f}'.format(i_episode, reward_episode))
        reward_list.append(reward_episode)

        if num_steps >= args.max_expert_state_num:
            break


main_loop()
print('Mean return:', sum(reward_list)/len(reward_list))
expert_traj = np.stack(expert_traj)

traj_saving_path = os.path.join(assets_dir(), 'expert_traj/TRPO/{}_expert_traj_50.p'.format(args.env_name))           
pickle.dump((expert_traj, running_state), open(traj_saving_path, 'wb'))
