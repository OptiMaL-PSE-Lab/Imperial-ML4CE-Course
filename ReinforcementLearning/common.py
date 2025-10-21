import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

'''
File containing common functions shared by the algorithms that model the policy with a neural network
'''

class PolicyNetwork(torch.nn.Module):
    '''
    MLP that takes environment state as input and outputs the mean value of each action.

    Assumption: actions follow independent normal distributions.
    '''
    def __init__(self, input_size, output_size, h1_size = 128, h2_size = 64): 
        super(PolicyNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, h1_size)
        self.fc2 = torch.nn.Linear(h1_size, h2_size)
        self.fc3 = torch.nn.Linear(h2_size, output_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x

class DiscretePolicyNetwork(torch.nn.Module):
    '''
    MLP that takes environment state as input and outputs the probability of taking each action
    '''
    def __init__(self, input_size, output_size, h1_size = 128, h2_size = 64): 
        super(DiscretePolicyNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, h1_size)
        self.fc2 = torch.nn.Linear(h1_size, h2_size)
        self.fc3 = torch.nn.Linear(h2_size, output_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.softmax(self.fc3(x),dim=-1)
        return x
    
def reward_fcn(policy_net, env, num_episodes=10, demand=None):
    '''
    Runs a series of episodes and computes the average total return.

    Arguments:
    - policy_net --> Neural network that predicts the optimal action given the state 
    - env --> Instance of MESCEnv environment
    - num_episodes --> Number of runs or episodes to estimate the average return (optional, default: 10)
    - demand --> List of scenario sets, containing realizations of customers' demand for each time step of each episode (optional, default: None)

    Returns:
    - mean_reward --> Average reward across runs
    - std_reward --> Standar deviation across runs
    '''
    # Input checking
    assert num_episodes > 0, "Number of episodes must be greater than 0"
    
    # Fix customer demand (if provided)
    env.demand_dataset = demand

    # Initialize buffer list to store results of each run
    reward_list = []

    # Run each episode and compute total undiscounted reward
    for i in range(num_episodes):
        # Reset environment before each episode
        env.reset()
        state = env.state
        episode_terminated = False
        # Initialize reward counter
        total_reward = 0
        
        while episode_terminated == False:
            # Sample action
            if isinstance(policy_net, PolicyNetwork):
                action_mean = policy_net(torch.FloatTensor(state)) 
                # TODO: Add covariance matrix and sample action from MultivariateNormal distribution
                action = np.fix(action_mean.detach().numpy())
            elif isinstance(policy_net, DiscretePolicyNetwork):
                action_probs = policy_net(torch.FloatTensor(state))
                dist = torch.distributions.Categorical(action_probs)
                action_idx = dist.sample().detach().numpy()
                action = np.array(list(env.action_lookup.keys())[action_idx])
            
            # Interact with the environment to get reward and next state
            state , reward, episode_terminated, _ = env.step(action)
            total_reward += reward
            
        reward_list.append(total_reward)

    # Compute mean and standard deviation
    mean_reward = np.mean(reward_list)
    std_reward = np.std(reward_list)
    
    return mean_reward, std_reward

def evaluate_policy(policy_net, env, test_demand_dataset):
    reward_list = []
    for demand in test_demand_dataset:
        reward, _ = reward_fcn(policy_net, env, num_episodes=1, demand=demand)
        reward_list.append(reward)
    return reward_list