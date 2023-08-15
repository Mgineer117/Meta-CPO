import pickle
import safety_gymnasium as gym
import numpy as np
import time
from gym import utils
from utils import *


path = "simulator/model.p"
policy, _, _, running_state, args = pickle.load(open(path, "rb"))
running_state.fix = True
dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    print('using gpu')
    torch.cuda.set_device(args.gpu_index)

#env = gym.make(args.env_name, render_mode = "human")

i = 0
env = gym.make("SafetyPointStbutton1-v0", render_mode='human')    
print('state dim: ', env.observation_space.shape[0])
print('action dim: ', env.action_space.shape[0])
while i < 100:
    seed = np.random.randint(1,1000)
    #seed = 1
    state, info = env.reset(seed = seed)
    state = running_state(state)
    env_reward_episode = 0
    env_cost_episode = 0
    t1 = time.time()
    for j in range(500):    
        state_var = tensor(state).unsqueeze(0)

        with torch.no_grad():
            if args.env_name == "CartPole-v0" or args.env_name == "CartPole-v1" or args.env_name == "MountainCar-v0" or args.env_name == "LunarLander-v2":
                action = policy.select_action(state_var)[0].numpy()
            else:
                action = policy(state_var)[0][0].numpy()

        #action = int(action) if policy.is_disc_action else action.astype(np.float64)
        next_state, reward, cost, done, truncated, _= env.step(action)
        
        env_reward_episode += reward
        env_cost_episode += cost * (0.99**j)
        next_state = running_state(next_state)

        if done or truncated:
            break

        state = next_state
    t2 = time.time()
    print("return: ", env_reward_episode, "cost: ", env_cost_episode, "time: ", t2-t1, "with step: ", j)
    
    i += 1

