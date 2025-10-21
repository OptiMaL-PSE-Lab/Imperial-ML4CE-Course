from ML4CE_RL_environment import MESCEnv
import numpy as np 
from scipy.optimize import minimize, basinhopping
import matplotlib.pyplot as plt

class HeuristicPolicy:
    def __init__(self, initial_policy=None):
        self.policy_param = initial_policy
        pass
    
    def set_initial_policy(self, env):
        initial_policy = np.ones(env.action_space.shape[0]*2,dtype=env.retailers[-1].I.dtype)
        initial_policy[1::2] = [retailer.order_quantity_limit for retailer in env.retailers] + [dc.order_quantity_limit for dc in env.DCs]
        initial_policy[0::2] = (1/3) * initial_policy[1::2] 
        self.policy_param = initial_policy

    def optimize_policy(self, env, objective_fcn, method='Powell', bounds=None, options=None, callback=None):

        if self.policy_param is None:
            self.set_initial_policy(env)

        if method == 'global':
            results = basinhopping(func = objective_fcn,
                                x0 = self.policy_param,
                                niter = 100 , 
                                minimizer_kwargs=  {'method': 'L-BFGS-B',
                                                    'args': (self.policy_fcn, env)},
                                callback= callback
                                )
        else:
            results = minimize(fun = objective_fcn,
                        x0 = self.policy_param,
                        args = (self.policy_fcn, env),
                        method = method,
                        bounds = bounds ,
                        tol = 1e-8,
                        options = options,
                        callback= callback
                        )

        return results

    def reward_fcn(self, policy_param, policy_function, env, demand=None):
        '''
        Runs an episode and computes the negative of the reward function.
        Reward function = expected profit
        '''
        # rewards = []
        total_reward = 0
        episode_terminated = False
        
        if demand == None:
            env.reset()
        else:
            env.reset(demand)

        if any(policy_param[0::2] >= policy_param[1::2]):
            return 1e8
        else:
            while episode_terminated == False:
                action = policy_function(policy_param, env)
                state , reward, episode_terminated, _ = env.step(action)
                # rewards.append(reward)
                total_reward += reward
                
            # return -1/env.n_periods * np.sum(env.prob_per_scenario*rewards)
            return -1*total_reward

    def objective_fcn(self, policy_param, policy_function, env, demand=None):
        num_runs = 5
        total_reward_list = []
        for _ in range(num_runs):
            total_reward = self.reward_fcn(policy_param, policy_function, env, demand)
            total_reward_list.append(total_reward)
        return np.mean(total_reward_list) , np.std(total_reward_list)

    def policy_fcn(self, policy_param, env):
        # Check that given x0 is a valid policy
        assert len(policy_param) == (len(env.retailers)+len(env.DCs))*2, "(s,S) policy should match the number of entities*2. \nMismatch {} vs {}".format(len(policy_param), (len(env.retailers)+len(env.DCs))*2)
        
        # Compute the action (order quantity)
        action = np.zeros(env.action_space.shape[0], dtype=env.retailers[-1].I.dtype)
        order_quantity_limit = np.array([retailer.order_quantity_limit for retailer in env.retailers] + [dc.order_quantity_limit for dc in env.DCs])
        for i, state in enumerate(env.state[:-1]):
            if state <= policy_param[i*2]:
                action[i] = np.minimum(policy_param[i*2+1] - state , order_quantity_limit[i])
            else:
                action[i] = 0

        return action.astype(env.retailers[-1].I.dtype)
    
    def evaluate_policy(self, env, test_demand_dataset):
        '''
        Evaluates a given policy in the environment.
        '''
        reward_list= []

        for demand in test_demand_dataset:
            env.demand_dataset = demand
            env.reset()
            reward_list.append(-1*self.reward_fcn(self.policy_param, self.policy_fcn , env))

        return reward_list

class Optimizer:
    def __init__(self, function, env):
        self.f      = function # actual objective function
        self.env    = env

        # Required when optimization algorithm iterations are tracked
        self.buffer_x      = [] # store all evaluations in one iteration
        self.buffer_f      = [] # store all evaluations in one iteration
        self.buffer_std    = [] # store all evaluations in one iteration
        self.iter          = 0
        self.tracking_x    = []
        self.tracking_f    = []
        self.tracking_std  = []
        self.best_f        = 1e8
        self.best_x        = None
        self.best_std      = None
    
    def calculate_reward(self, policy_param, policy_fcn, env):
        reward = self.f(policy_param, policy_fcn, env)
        self.buffer_x.append(policy_param)
        self.buffer_f.append(reward[0])
        self.buffer_std.append(reward[-1])

        return reward[0]
    
    def callback(self, xk):
        # Retrieve the best values for the current interation
        idx = np.where(np.all(xk == self.buffer_x, axis = 1))[0].tolist()
        self.tracking_f.append(min(np.array(self.buffer_f)[idx]))
        self.tracking_x.append(xk)
        self.tracking_std.append(np.mean(np.array(self.buffer_std)[idx]))

        # Re-initialize buffers
        self.buffer_f = []
        self.buffer_x = []
        self.buffer_std = []
        
        # Update iteration counter
        self.iter += 1

    def callback_global(self, xk, f, accept):
        self.tracking_f.append(f)
        self.tracking_x.append(xk)

    def get_best_solution(self):
        self.best_f = min(self.tracking_f)
        idx = np.where(self.best_f==self.tracking_f)[0]
        self.best_x = self.tracking_x[idx[-1]]
        self.best_std = self.tracking_std[idx[-1]]

    def plot_learning_curve(self):
        reward_arr = -1 * np.array(self.tracking_f)
        std_arr = -1* np.array(self.tracking_std)
        episodes = list(range(1, len(reward_arr) + 1))
        fig = plt.figure(figsize=(7,5))
        # Plot mean line
        plt.plot(episodes, reward_arr, label='Mean return', color='blue')
        # Plot shaded area for std
        plt.fill_between(episodes, 
                         reward_arr - std_arr, 
                         reward_arr + std_arr, 
                         color='blue', alpha=0.2, label='±1 std')
        plt.xlabel('Iterations')
        plt.ylabel('Reward (MU)') # TODO update monetary units 