from os import path
import safety_gymnasium as gym

def hline():
    print('==============================================')

def create_envs(args):
    envs = []
    print('\n')
    hline()
    print(f'{1} Environement generated... with parameters:')
    hline()
    env = gym.make(args.env_name + str(0) + '-v0')
    envs.append(env)
    for i in range(1, args.env_num+1):
        if i != args.env_num:
            hline()
            print(f'{i+1} Environement generated... with parameters:')
            hline()
            env = gym.make(args.env_name + str(1) + '-v0')
        else:
            hline()
            print('Test Environement generated... with parameters:')
            hline()
            env = gym.make(args.env_name + str(2) + '-v0')
            
        
        envs.append(env)
    
    return envs
def assets_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../assets'))

