import multiprocessing
from utils.replay_memory import Memory
from utils.torch import *
import math
import time
import pdb

class bcolors:
    MAGENTA = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    YELLOW = '\033[93m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RED = '\033[91m'
    WHITE = '\033[97m'
    BLACK = '\033[90m'
    DEFAULT = '\033[99m'

def collect_samples(pid, queue, env, policy, custom_reward,
                    mean_action, render, running_state, min_batch_size, seed):
    torch.randn(pid)
    log = dict()
    memory = Memory()
    num_steps = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    total_c_reward = 0
    min_c_reward = 1e6
    max_c_reward = -1e6
    env_total_reward = 0
    env_total_cost = 0
    num_episodes = 0
    reward_episode_list = []
    c_reward_episode_list = []
    env_reward_episode_list = []
    """
    #randomize environment seed
    seed = np.random.randint(1,1000)
    env.seed(seed)
    torch.manual_seed(seed)
    """
    while num_steps < min_batch_size:
        state, info = env.reset(seed=seed)
        if running_state is not None:
            state = running_state(state)
        
        reward_episode = 0
        env_reward_episode = 0
        env_cost_episode = 0
        reward_episode_list_1 = []
        env_reward_episode_list_1 = []
        
        for t in range(10000):
            state_var = tensor(state).unsqueeze(0)
            with torch.no_grad():
                if mean_action:
                    action = policy(state_var)[0][0].numpy()
                    #print(action)
                else:
                    # Stochastic action
                    action = policy.select_action(state_var)[0].numpy()
    
            #print(action)
            action = int(action) if policy.is_disc_action else action.astype(np.float64)
            next_state, reward, cost, done, truncated, _= env.step(action)
            #print(cost)
            """if t % 500 == 0:
                print(reward)"""
            env_reward_episode += reward
            env_cost_episode += cost
            env_reward_episode_list_1.append(reward)

            if running_state is not None:
                next_state = running_state(next_state)

            mask = 0 if done else 1

            memory.push(state, action, mask, next_state, reward, cost)
            if render:
                env.render()
            if done or truncated:
                break

            state = next_state

        # log stats
        num_steps += (t + 1)
        num_episodes += 1
        env_reward_episode_list.append(env_reward_episode)
        env_total_reward += env_reward_episode
        env_total_cost += env_cost_episode
        min_reward = min(min_reward, env_reward_episode)
        max_reward = max(max_reward, env_reward_episode)

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['env_total_reward'] = env_total_reward
    log['env_total_cost'] = env_total_cost
    log['env_avg_reward'] = env_total_reward / num_episodes
    log['env_avg_cost'] = env_total_cost / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    log['env_reward_ep_list'] = env_reward_episode_list

    if queue is not None:
        queue.put([pid, memory, log])
    else:
        return memory, log


def merge_log(log_list):
    log = dict()
    total_rewards_episodes = []

    # merge env reward 
    log['env_total_reward'] = sum([x['env_total_reward'] for x in log_list])
    log['env_total_cost'] = sum([x['env_total_cost'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    log['env_avg_reward'] = log['env_total_reward'] / log['num_episodes']
    log['env_avg_cost'] = log['env_total_cost'] / log['num_episodes']
    for x in log_list:
        b = x['env_reward_ep_list']
        total_rewards_episodes += b
    log['total_rewards_episodes'] = total_rewards_episodes
    #print('np.array(log_total_rewards_episodes)', np.array(log['total_rewards_episodes']))

    """std deviation of env rewards in one iteration"""
    #print(total_rewards_episodes)
    #print(log['env_avg_reward'])
    reward_episode_list_array = np.array(total_rewards_episodes) - log['env_avg_reward']
    #print(reward_episode_list_array)

    reward_episode_list_array = np.square(reward_episode_list_array)
    #print(reward_episode_list_array)

    reward_episode_list_sum = np.sum(reward_episode_list_array)
    #print(reward_episode_list_sum)

    reward_episode_list_variance = reward_episode_list_sum / log['num_episodes']
    #print(reward_episode_list_variance)

    reward_episode_list_std = np.sqrt(reward_episode_list_variance)
    #print(reward_episode_list_std)
    log['std_reward']  = reward_episode_list_std
    
    return log


class Agent:

    def __init__(self, env, policy, device, custom_reward=None,
                 mean_action=False, render=False, running_state=None, num_threads=1):
        self.env = env
        self.policy = policy
        self.device = device
        self.custom_reward = custom_reward
        self.mean_action = mean_action
        self.running_state = running_state
        self.render = render
        self.num_threads = num_threads

