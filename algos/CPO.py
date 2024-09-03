import numpy as np
import scipy.optimize
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import pdb
import multiprocessing
import time
import torch


from utils import *
from core.common  import estimate_advantages, estimate_constraint_value

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

def collect_samples(pid, queue, env, policy,
                    mean_action, running_state, min_batch_size, horizon, seed):
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

    if queue is not None:
        # this is to apply different stochasity
        # in replicate of main process
        torch.manual_seed(seed + pid)
    
    while num_steps < min_batch_size:
        t0 = time.time()
        state, info = env.reset(seed=seed)
        t1 = time.time()
        if running_state is not None:
            state = running_state(state)
        
        reward_episode = 0
        env_reward_episode = 0
        env_cost_episode = 0
        reward_episode_list_1 = []
        env_reward_episode_list_1 = []
        t2 = time.time()
        for t in range(horizon):
            state_var = tensor(state).unsqueeze(0)
            with torch.no_grad():
                if mean_action:
                    action = policy(state_var)[0][0].numpy()
                else:
                    # Stochastic action
                    action = policy.select_action(state_var)[0].numpy()
    
            action = int(action) if policy.is_disc_action else action.astype(np.float64)
            next_state, reward, cost, done, truncated, _= env.step(action)

            env_reward_episode += reward
            env_cost_episode += cost
            env_reward_episode_list_1.append(reward)

            if running_state is not None:
                next_state = running_state(next_state)

            mask = 0 if done else 1

            memory.push(state, action, mask, next_state, reward, cost)

            if done or truncated:
                break

            state = next_state
        t3 = time.time()
        #print("reset time: ", t1- t0, "local: ", t3-t2, 'step: ', t)
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

    """std deviation of env rewards in one iteration"""
    reward_episode_list_array = np.array(total_rewards_episodes) - log['env_avg_reward']
    reward_episode_list_array = np.square(reward_episode_list_array)
    reward_episode_list_sum = np.sum(reward_episode_list_array)
    reward_episode_list_variance = reward_episode_list_sum / log['num_episodes']
    reward_episode_list_std = np.sqrt(reward_episode_list_variance)
    log['std_reward']  = reward_episode_list_std
    
    return log

def trpo_problem(param_size, dtype):
    x = cp.Variable(param_size)

    g = cp.Parameter(param_size)
    max_kl = cp.Parameter()

    objective = cp.Minimize(g.T @ x)
    constraints = [0.5 * cp.sum_squares(x) <= max_kl]

    problem = cp.Problem(objective, constraints)
    cvxpylayer = CvxpyLayer(problem, parameters=[g, max_kl], variables=[x])

    return cvxpylayer

def cpo_problem(param_size, dtype):
    x = cp.Variable(param_size)

    g = cp.Parameter(param_size)
    max_kl = cp.Parameter()

    a = cp.Parameter(param_size)
    max_constraint = cp.Parameter()

    objective = cp.Minimize(g.T @ x)
    constraints = [0.5 * cp.sum_squares(x) <= max_kl,
                   max_constraint + a.T @ x <= 0]

    problem = cp.Problem(objective, constraints)
    cvxpylayer = CvxpyLayer(problem, parameters=[g, max_kl, a, max_constraint], variables=[x])

    return cvxpylayer

class CPO:
    def __init__(self, envs, policy_net, value_net, cost_net, args, mean_action=False, running_state=None):
        self.envs = envs
        self.state_dim = envs[0].observation_space.shape[0]
        self.action_dim = envs[0].action_space.shape[0]
        self.running_state = running_state
        self.mean_action = mean_action

        self.args = args

        self.min_batch = self.args.min_batch_size
        self.num_threads = self.args.num_threads

        self.dtype = args.dtype
        self.device = args.device

        self.policy_net = policy_net
        self.value_net = value_net
        self.cost_net = cost_net

        self.param_size = sum(p.numel() for p in policy_net.parameters())
        self.cpo_problem = cpo_problem(self.param_size, self.dtype)
        self.trpo_problem = trpo_problem(self.param_size, self.dtype)

        self.loss_fn = torch.nn.MSELoss()
        # value network is usually set higher lr than policy since it is a bottleneck in learning
        if args.bfgs_iter_num is not None:
            self.value_optimizer = torch.optim.LBFGS(self.value_net.parameters(), lr=args.critic_lr, max_iter=args.bfgs_iter_num)
            self.cost_optimizer = torch.optim.LBFGS(self.cost_net.parameters(), lr=args.critic_lr, max_iter=args.bfgs_iter_num)
        else:
            self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=args.critic_lr)
            self.cost_optimizer = torch.optim.Adam(self.cost_net.parameters(), lr=args.critic_lr)

    def line_search(self, model, reward_f, cost_f, x, fullstep, expected_reward_improve_full, expected_cost_improve_full, max_backtracks=10, accept_ratio=0.1):
        reward_fval = reward_f(True).item()
        cost_fval = cost_f(True).item()
        
        for stepfrac in [.5**x for x in range(max_backtracks)]:
            x_new = x + stepfrac * fullstep
            set_flat_params_to(model, x_new)

            reward_fval_new = reward_f(True).item()
            cost_fval_new = cost_f(True).item()

            actual_reward_improve = reward_fval - reward_fval_new  
            actual_cost_improve = cost_fval - cost_fval_new

            expected_reward_improve = expected_reward_improve_full * stepfrac
            expected_cost_improve = expected_cost_improve_full * stepfrac

            r_ratio = actual_reward_improve / expected_reward_improve
            c_ratio = actual_cost_improve / expected_cost_improve
            if torch.norm(stepfrac * fullstep) <= self.args.max_kl and r_ratio > accept_ratio and c_ratio > accept_ratio:
                return True, x_new
        return False, x
    
    def collect_samples(self, env, seed):
        t_start = time.time()
        to_device(torch.device('cpu'), self.policy_net)
        thread_batch_size = int(math.floor(self.min_batch / self.num_threads))
        queue = multiprocessing.Queue()
        workers = []
        for i in range(self.num_threads-1):
            worker_args = (i+1, queue, env, self.policy_net, self.mean_action, self.running_state, thread_batch_size, self.args.time_horizon, seed)
            workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))
        for worker in workers:
            worker.start()
        memory, log = collect_samples(0, None, env, self.policy_net, 
                                      self.mean_action, self.running_state, 
                                      thread_batch_size, self.args.time_horizon, seed)
        

        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        for _ in workers:
            pid, worker_memory, worker_log = queue.get()
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log
        for worker_memory in worker_memories:
            memory.append(worker_memory)
        batch = memory.sample()
        if self.num_threads > 1:
            log_list = [log] + worker_logs
            log = merge_log(log_list)
        to_device(self.device, self.policy_net)
        t_end = time.time()
        log['sample_time'] = t_end - t_start
        log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        return batch, log

    def cpo_step(self, batch, use_fim=False):
        states = torch.from_numpy(np.stack(batch.state)[:self.args.max_batch_size]).to(self.dtype).to(self.device) #[:args.batch_size]
        costs =  torch.from_numpy(np.stack(batch.cost)[:self.args.max_batch_size]).to(self.dtype).to(self.device) #[:args.batch_size]
        actions = torch.from_numpy(np.stack(batch.action)[:self.args.max_batch_size]).to(self.dtype).to(self.device)
        rewards = torch.from_numpy(np.stack(batch.reward)[:self.args.max_batch_size]).to(self.dtype).to(self.device)
        masks = torch.from_numpy(np.stack(batch.mask)[:self.args.max_batch_size]).to(self.dtype).to(self.device)

        reward_values = self.value_net(states)
        cost_values = self.cost_net(states)

        """get advantage estimation from the trajectories"""
        reward_advantages, reward_returns = estimate_advantages(rewards, masks, reward_values.clone().detach(), self.args.gamma, self.args.tau, self.device)
        cost_advantages, cost_returns = estimate_advantages(costs, masks, cost_values.clone().detach(), self.args.gamma, self.args.tau, self.device)
        constraint_value = estimate_constraint_value(costs, masks, self.args.gamma, self.device)
        
        reward_value_loss = self.loss_fn(reward_values, reward_returns)
        cost_value_loss = self.loss_fn(cost_values, cost_returns)

        if self.args.bfgs_iter_num is not None:
            def r_closure():
                self.value_optimizer.zero_grad()
                r_pred = self.value_net(states)
                v_loss = self.loss_fn(r_pred, reward_returns)
                for param in self.value_net.parameters():
                    v_loss += param.pow(2).sum() * self.args.l2_reg
                v_loss.backward()
                # gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)
                return v_loss

            def c_closure():
                self.cost_optimizer.zero_grad()
                c_pred = self.cost_net(states)
                c_loss = self.loss_fn(c_pred, cost_returns)
                for param in self.cost_net.parameters():
                    c_loss += param.pow(2).sum() * self.args.l2_reg
                c_loss.backward()
                # gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.cost_net.parameters(), max_norm=1.0)
                return c_loss
            
            self.value_optimizer.step(r_closure)
            self.cost_optimizer.step(c_closure)
        else:
            self.value_optimizer.zero_grad()
            reward_value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)
            self.value_optimizer.step()

            self.cost_optimizer.zero_grad()
            cost_value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.cost_net.parameters(), max_norm=1.0)
            self.cost_optimizer.step()

        """update policy"""
        with torch.no_grad():
            fixed_log_probs = self.policy_net.get_log_prob(states, actions)

        def get_reward_loss(volatile=False):
            with torch.set_grad_enabled(not volatile):
                log_probs = self.policy_net.get_log_prob(states, actions)
                action_loss = -reward_advantages * torch.exp(log_probs - fixed_log_probs)
                return action_loss.mean()
            
        """define the cost loss function for TRPO"""
        def get_cost_loss(volatile=False):
            with torch.set_grad_enabled(not volatile):
                log_probs = self.policy_net.get_log_prob(states, actions)
                action_loss = cost_advantages * torch.exp(log_probs - fixed_log_probs)
                return action_loss.mean()

        reward_loss = get_reward_loss()
        cost_loss = get_cost_loss()

        """define the loss function for TRPO"""
        reward_grads = torch.autograd.grad(reward_loss, self.policy_net.parameters())
        loss_reward_grad = torch.cat([grad.view(-1) for grad in reward_grads])

        cost_grads = torch.autograd.grad(cost_loss, self.policy_net.parameters())
        loss_cost_grad = torch.cat([grad.view(-1) for grad in cost_grads])

        """Gradient normalization"""
        if self.args.grad_norm:
            loss_reward_grad = loss_reward_grad / torch.norm(loss_reward_grad)
            loss_cost_grad = loss_cost_grad / torch.norm(loss_cost_grad)   

        """DP"""
        g = loss_reward_grad.clone().detach()
        a = loss_cost_grad.clone().detach()

        max_kl = torch.tensor(self.args.max_kl, dtype=self.dtype).to(self.device)
        max_constraint = torch.tensor(self.args.max_constraint, dtype=self.dtype)

        b = (constraint_value[0] - max_constraint).to(self.device)
        if self.args.anneal:
            self.args.max_constraint -= self.args.annealing_factor * self.args.max_constraint
            
        try:
            step, = self.cpo_problem(g, max_kl, a, b)
            #print('step mean: ', abs(step).mean())
            prev_params = get_flat_params_from(self.policy_net)
            expected_reward_improve = -loss_reward_grad.dot(step)
            expected_cost_improve = -loss_cost_grad.dot(step)
            _, new_params = self.line_search(self.policy_net, get_reward_loss, get_cost_loss, prev_params, step,
                                              expected_reward_improve, expected_cost_improve)

            set_flat_params_to(self.policy_net, new_params)
        except:
            step, = self.trpo_problem(a, max_kl)
            print('recover mean: ', abs(step).mean())

            prev_params = get_flat_params_from(self.policy_net)
            expected_reward_improve = -loss_reward_grad.dot(step)
            expected_cost_improve = -loss_cost_grad.dot(step)
            _, new_params = self.line_search(self.policy_net, get_reward_loss, get_cost_loss, prev_params, step,
                                              expected_reward_improve, expected_cost_improve)
            set_flat_params_to(self.policy_net, new_params)

            return reward_value_loss.item(), cost_value_loss.item(), reward_loss, cost_loss

        return reward_value_loss.item(), cost_value_loss.item(), reward_loss, cost_loss

    def meta_test(self, writer):
        print("Start meta-testing")
        meta_avg_cost = []
        for m_iter in range(self.args.meta_iter_num):
            sample_time = 0
            seed = np.random.randint(1,2**20)
            batch, log = self.collect_samples(self.envs[-1], seed)
            
            sample_time += log['sample_time']

            t1 = time.time()
            _, _, _, _ = self.cpo_step(batch)
            t2 = time.time()

            self.running_state.fix = True
            self.mean_action=True
            
            seed = np.random.randint(1,2**20)
            eval_batch, eval_log = self.collect_samples(self.envs[-1], seed)
            sample_time += eval_log['sample_time']

            self.running_state.fix = False 
            self.mean_action=False

            # calculate values
            costs =  torch.from_numpy(np.stack(eval_batch.cost)[:self.args.max_batch_size]).to(self.dtype).to(self.device) #[:args.batch_size]
            masks = torch.from_numpy(np.stack(eval_batch.mask)[:self.args.max_batch_size]).to(self.dtype).to(self.device)
            eval_cost = estimate_constraint_value(costs, masks, self.args.gamma, self.device)[0].to(torch.device('cpu'))

            meta_avg_cost.append(eval_cost)

            print('{}\tT_sample {:.4f}  T_update {:.4f}\tC_avg/iter {:.2f}  Test_C_avg {:.2f}\tR_avg {:.2f}\tTest_R_avg {:.2f}\tTest_R_std {:.2f}'.format( 
            m_iter, sample_time, t2-t1, np.average(meta_avg_cost), eval_cost, log['env_avg_reward'], eval_log['env_avg_reward'], eval_log['std_reward']))   

            writer.add_scalar('MetaTesting/rewards', eval_log['env_avg_reward'], m_iter)
            writer.add_scalar('MetaTesting/costs', eval_cost, m_iter) 

            """clean up gpu memory"""
            torch.cuda.empty_cache()

        writer.close()
        return meta_avg_cost
      
    def train_CPO(self, writer, save_info_obj):
        # variables and lists for recording losses
        v_loss_list = []
        c_loss_list = []
        reward_loss_list = []
        cost_loss_list = []
        
        # lists for dumping plotting data for agent
        rewards_std = []
        env_avg_reward = []
        num_of_steps = []
        num_of_episodes = []
        total_num_episodes = []
        total_num_steps = []
        tne = 0 #cummulative number of episodes
        tns = 0 #cummulative number of steps
        
        # lists for dumping plotting data for mean agent
        eval_avg_reward = []
        eval_avg_reward_std = []
        iter_for_best_avg_reward = None
        
        eval_avg_cost = []

        # for saving the best model
        best_avg_reward = 0
        best_std = 5
        
        if self.args.model_path is not None:
            total_iterations = self.args.max_iter_num - self.args.update_iter_num
            print('total iterations: ',self.args.max_iter_num,
               'updated iteration: ', self.args.update_iter_num,
               'remaining iteration: ', total_iterations)
        else: 
            self.args.update_iter_num = 0
        
        if self.args.is_meta_test:
            meta_avg_cost = self.meta_test(writer)
            return 

        for i_iter in range(self.args.update_iter_num, self.args.max_iter_num):
            """generate multiple trajectories that reach the minimum batch_size"""
            seed = np.random.randint(1,2**20)

            #self.mean_action = True
            batch, log = self.collect_samples(self.envs[0], seed)

            t0 = time.time()
            v_loss, c_loss, reward_loss, cost_loss = self.cpo_step(batch)
            t1 = time.time()

            # update lists for saving
            v_loss_list.append(v_loss)
            c_loss_list.append(c_loss)
            reward_loss_list.append(reward_loss)
            cost_loss_list.append(cost_loss)
            rewards_std.append(log['std_reward']) 
            env_avg_reward.append(log['env_avg_reward'])
            num_of_steps.append(log['num_steps'])
            num_of_episodes.append(log['num_episodes'])
            tne = tne + log['num_episodes']
            tns = tns + log['num_steps']
            total_num_episodes.append(tne)
            total_num_steps.append(tns)          

            # evaluate the current policy
            self.running_state.fix = True  #Fix the running state
            self.mean_action = True

            eval_batch, eval_log = self.collect_samples(self.envs[0], seed)

            self.running_state.fix = False
            self.mean_action = False

            # calculate values
            costs =  torch.from_numpy(np.stack(eval_batch.cost)[:self.args.max_batch_size]).to(self.dtype).to(self.device) #[:args.batch_size]
            masks = torch.from_numpy(np.stack(eval_batch.mask)[:self.args.max_batch_size]).to(self.dtype).to(self.device)
            eval_cost = estimate_constraint_value(costs, masks, self.args.gamma, self.device)[0].to(torch.device('cpu'))

            # update eval lists
            eval_avg_reward.append(eval_log['env_avg_reward'])
            eval_avg_reward_std.append(eval_log['std_reward'])
            eval_avg_cost.append(eval_cost)

            # update tensorboard summaries
            writer.add_scalar('CriticLoss/reward_value_loss', v_loss, i_iter)
            writer.add_scalar('CriticLoss/cost_value_loss', c_loss, i_iter)
            writer.add_scalar('PolicyLoss/reward_advantage_loss', reward_loss, i_iter)
            writer.add_scalar('PolicyLoss/cost_advantage_loss', cost_loss, i_iter)
            writer.add_scalar('Training/rewards', eval_log['env_avg_reward'], i_iter)  
            writer.add_scalar('Training/costs', eval_cost, i_iter)  
            writer.add_scalar('Training/reward_std', eval_log['std_reward'], i_iter)

            # print learning data on screen     
            if i_iter % self.args.log_interval == 0:
                print('{}\tT_sample {:.4f}  T_update {:.4f}\tC_avg/iter {:.2f}  Test_C_avg {:.2f}\tR_avg {:.2f}\tTest_R_avg {:.2f}\tTest_R_std {:.2f}'.format( 
                    i_iter, log['sample_time'], t1-t0, np.average(eval_avg_cost), eval_cost, log['env_avg_reward'], eval_log['env_avg_reward'], eval_log['std_reward']))              
            
            # save the best model
            if eval_log['env_avg_reward'] >= best_avg_reward and eval_log['std_reward'] <= best_std:
                print('Saving new best model !!!!')
                to_device(torch.device('cpu'), self.policy_net, self.value_net, self.cost_net)
                save_info_obj.save_models(self.policy_net, self.value_net, self.cost_net, self.running_state, self.args)
                to_device(self.device, self.policy_net, self.value_net, self.cost_net)
                best_avg_reward = eval_log['env_avg_reward']
                best_std = eval_log['std_reward']
                iter_for_best_avg_reward = i_iter+1  

            # save some intermediate models to sample trajectories from
            if self.args.save_intermediate_model > 0 and (i_iter+1) % self.args.save_intermediate_model == 0:
                to_device(torch.device('cpu'), self.policy_net, self.value_net, self.cost_net)
                save_info_obj.save_intermediate_models(self.policy_net, self.value_net, self.cost_net, self.running_state, self.args, i_iter)
                to_device(self.device, self.policy_net, self.value_net, self.cost_net)
            
            """clean up gpu memory"""
            torch.cuda.empty_cache()

        meta_avg_cost = self.meta_test(writer)
            
        # dump expert_avg_reward, num_of_steps, num_of_episodes
        save_info_obj.dump_lists(best_avg_reward, num_of_steps, num_of_episodes, total_num_episodes, 
                                 total_num_steps, rewards_std, env_avg_reward, v_loss_list, c_loss_list, reward_loss_list, cost_loss_list,
                                 eval_avg_reward, eval_avg_reward_std, eval_avg_cost, meta_avg_cost)
        print("Best iter: ", iter_for_best_avg_reward, 'Best eval Reward:', best_avg_reward, "best reward std:", best_std)
        return best_avg_reward, best_std, iter_for_best_avg_reward 