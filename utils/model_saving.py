import os
import numpy as np
import pickle

from datetime import date
today = date.today()

class save_info(object):
    def __init__(self, assets_dir, exp_num, exp_name, env_name):
        self.assets_dir = assets_dir
        self.experiment_num = 'exp-{}'.format(exp_num)
        #common path
        self.saving_path = 'learned_models/{}/{}-{}-{}'.format(exp_name, today, self.experiment_num, env_name)
        

    def create_all_paths(self):
        """create all the paths to save learned models/data"""
        #model saving path
        self.model_saving_path = os.path.join(self.assets_dir, self.saving_path, 'model.p') 
        self.env_saving_path = os.path.join(self.assets_dir, self.saving_path, 'env.p') 
        if not os.path.exists(os.path.dirname(self.model_saving_path)):
            try:
                os.makedirs(os.path.dirname(self.model_saving_path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        
        #intermediate model saving path
        self.intermediate_model_saving_path = os.path.join(self.assets_dir, self.saving_path, 'intermediate_model/') 
        if not os.path.exists(os.path.dirname(self.intermediate_model_saving_path)):
            try:
                os.makedirs(os.path.dirname(self.intermediate_model_saving_path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        
        #avg reward saving path    
        self.avg_reward_saving_path = os.path.join(self.assets_dir, self.saving_path, 'avg_reward.p')
        
        # num of steps saving path
        self.num_of_steps_saving_path = os.path.join(self.assets_dir, self.saving_path, 'num_of_steps.p') 
        
        # num of episodes saving path
        self.num_of_episodes_saving_path = os.path.join(self.assets_dir, self.saving_path, 'num_of_episodes.p')
        
        # total num of episodes saving path
        self.total_num_of_episodes_saving_path = os.path.join(self.assets_dir, self.saving_path, 'total_num_of_episodes.p')
        
        # total steps num saving path
        self.total_num_of_steps_saving_path = os.path.join(self.assets_dir, self.saving_path, 'total_num_of_steps.p')
        
        # total num of steps saving path
        self.rewards_std_saving_path = os.path.join(self.assets_dir, self.saving_path, 'rewards_std.p')
        
        # total num of steps saving path
        self.env_avg_reward_saving_path = os.path.join(self.assets_dir, self.saving_path, 'env_avg_reward.p')
        
        # v loss saving path
        self.true_v_loss_list_saving_path = os.path.join(self.assets_dir, self.saving_path, 'true_v_loss_list.p') 
        
        # decayed v loss saving path
        self.decayed_v_loss_list_saving_path = os.path.join(self.assets_dir, self.saving_path, 'decayed_v_loss_list.p')
        
        # c loss saving path
        self.c_loss_list_saving_path = os.path.join(self.assets_dir, self.saving_path, 'p_loss_list.p')

        # reward loss saving path
        self.reward_loss_list_saving_path = os.path.join(self.assets_dir, self.saving_path, 'c_loss_list.p')

        # cost loss saving path
        self.cost_loss_list_saving_path = os.path.join(self.assets_dir, self.saving_path, 'c_loss_list.p')
        
        # evaluation average reward 
        self.eval_avg_R_saving_path = os.path.join(self.assets_dir, self.saving_path, 'eval_avg_R.p')
        
        # evaluation average reward std
        self.eval_avg_R_std_saving_path = os.path.join(self.assets_dir, self.saving_path, 'eval_avg_R_std.p')

        # evaluation average cost 
        self.eval_avg_C_saving_path = os.path.join(self.assets_dir, self.saving_path, 'eval_avg_C.p')
        
        # evaluation average cost 
        self.meta_avg_C_saving_path = os.path.join(self.assets_dir, self.saving_path, 'meta_avg_C.p')
        
        
    def dump_lists(self, avg_reward, num_of_steps, num_of_episodes, total_num_episodes, total_num_steps, rewards_std, env_avg_reward, v_loss_list, c_loss_list, reward_loss_list, cost_loss_list, eval_avg_reward, eval_avg_reward_std, eval_avg_cost, meta_avg_cost):
        
        # dump expert_avg_reward, num_of_steps, num_of_episodes
        pickle.dump(avg_reward,
                    open(os.path.join(self.assets_dir, self.avg_reward_saving_path),'wb'))
        pickle.dump(num_of_steps,
                    open(os.path.join(self.assets_dir, self.num_of_steps_saving_path),'wb'))
        pickle.dump(num_of_episodes,
                    open(os.path.join(self.assets_dir, self.num_of_episodes_saving_path), 'wb'))
        pickle.dump(total_num_episodes,
                    open(os.path.join(self.assets_dir, self.total_num_of_episodes_saving_path), 'wb'))
        pickle.dump(total_num_steps,
                    open(os.path.join(self.assets_dir, self.total_num_of_steps_saving_path), 'wb'))
        pickle.dump(rewards_std,
                    open(os.path.join(self.assets_dir, self.rewards_std_saving_path), 'wb'))
        pickle.dump(env_avg_reward,
                    open(os.path.join(self.assets_dir, self.env_avg_reward_saving_path), 'wb'))
        pickle.dump(v_loss_list,
                    open(os.path.join(self.assets_dir, self.true_v_loss_list_saving_path), 'wb'))
        pickle.dump(c_loss_list,
                    open(os.path.join(self.assets_dir, self.c_loss_list_saving_path), 'wb'))
        pickle.dump(reward_loss_list,
                    open(os.path.join(self.assets_dir, self.reward_loss_list_saving_path), 'wb'))
        pickle.dump(cost_loss_list,
                    open(os.path.join(self.assets_dir, self.cost_loss_list_saving_path), 'wb'))
        pickle.dump(eval_avg_reward,
                    open(os.path.join(self.assets_dir, self.eval_avg_R_saving_path), 'wb'))
        pickle.dump(eval_avg_reward_std,
                    open(os.path.join(self.assets_dir, self.eval_avg_R_std_saving_path), 'wb'))
        pickle.dump(eval_avg_cost,
                    open(os.path.join(self.assets_dir, self.eval_avg_C_saving_path), 'wb'))
        pickle.dump(meta_avg_cost,
                    open(os.path.join(self.assets_dir, self.meta_avg_C_saving_path), 'wb'))
        
    def save_envs(self, envs):
        pickle.dump(envs, open(os.path.join(self.assets_dir, self.env_saving_path), 'wb'))
    def save_models(self, policy_net, value_net, cost_net, running_state, args):
        pickle.dump((policy_net, value_net, cost_net, running_state, args), open(os.path.join(self.assets_dir, self.model_saving_path), 'wb'))
        
    def save_intermediate_models(self, policy_net, value_net, cost_net, running_state, args, i_iter):    
        pickle.dump((policy_net, value_net, cost_net, running_state, args), open(os.path.join(self.assets_dir, self.intermediate_model_saving_path, 'model_iter_{}.p'.format(i_iter+1)), 'wb'))
