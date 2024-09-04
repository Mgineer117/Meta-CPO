import argparse

def parse_all_arguments():
        
    parser = argparse.ArgumentParser() #description='Running {}'.format(algo_name))
    
    # Basic agruments 
    parser.add_argument('--algo-name', default="CPOMeta", metavar='G',
                       help='algorithm name')
    parser.add_argument('--exp-num', default="1", metavar='G',
                        help='Experiment number for today (default: 1)')
    parser.add_argument('--exp-name', default="Exp-1", metavar='G',
                        help='Experiment name')
    parser.add_argument('--env-name', default="SafetyCarStcircle", metavar='G',
                        help='name of the environment to run')
    parser.add_argument('--env-num', type=int, default=3, metavar='G',
                        help='number of environments')
    
    # update with prev parameters
    parser.add_argument('--model-path', metavar='G', #default="post_training/",
                        help='path of pre-trained model')
    parser.add_argument('--is-meta-test', default=False,
                        help='do the meta-testing after training')
    parser.add_argument('--update-iter-num', metavar='N', type=int, default=0, 
                        help='iter num for meta-test')

    # Learning rates and regularizations
    parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                        help='log std for the policy (default: -0.0)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                        help='gae (default: 0.95)')
    parser.add_argument('--l2-reg', type=float, default=1e-5, metavar='G',
                        help='l2 regularization of value function (default: 1e-4)')
    parser.add_argument('--bfgs-iter-num', type=int, default=25, metavar='G',
                        help='if it is set to None, Adam is used (default: 10)')
    parser.add_argument('--policy-lr', type=float, default=1e-1, metavar='G',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--critic-lr', type=float, default=1e-3, metavar='G',
                        help='learning rate (default: 1e-3)')
    
    # GPU index, multi-threading and seeding
    parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
    parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                        help='number of threads for multiprocessing (default: 4)')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 0)')
    
    # batch size and iteration number
    parser.add_argument('--min-batch-size', type=int, default=3000, metavar='N',
                        help='minimal batch size per PPO update (default: 3000)')
    parser.add_argument('--max-batch-size', type=int, default=3000, metavar='N',
                        help='maximum batch size per PPO update (default: 3000)')
    parser.add_argument('--time-horizon', type=int, default=500, metavar='N',
                        help='time step for one horizon (default: 500)')
    parser.add_argument('--max-iter-num', type=int, default=1000, metavar='N',
                        help='maximal number of main iterations (default: 500)')
    parser.add_argument('--meta-iter-num', type=int, default=50, metavar='N',
                        help='maximal number of main iterations (default: 100)')                       
    parser.add_argument('--local-num', type=int, default=3, metavar='N',
                        help='maximal number of main iterations (default: 3)')
    
    # logging and saving models
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--save-model-interval', type=int, default=2, metavar='N',
                        help="interval between saving model (default: 0, means don't save)")
    parser.add_argument('--save-intermediate-model', type=int, default=2, metavar='N',
                        help="intermediate model saving interval (default: 0, means don't save)")

    parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
    parser.add_argument('--max-constraint', type=float, default=10, metavar='G',
                    help='max constraint value (default: 10 ~ 20)')
    parser.add_argument('--annealing_factor', type=float, default=1e-4, metavar='G',
                    help='annealing factor of constraint (default: 1e-5)')
    parser.add_argument('--anneal', default=True,
                    help='Should the constraint be annealed or not')
    parser.add_argument('--grad-norm', default=True,
                    help='Should the norm of policy gradient be taken (default: False)')    
    
    return parser.parse_args()

