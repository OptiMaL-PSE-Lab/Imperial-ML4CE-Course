import numpy as np 
import torch
import time
from tqdm import tqdm

from common import PolicyNetwork
from utils import setup_model_saving

def REINFORCE_alg(  env, *, 
                    max_episodes = 200 , 
                    max_time = 60. ,
                    lr_policy_net= 1e-3, 
                    lr_value_net = 5e-3, 
                    discount_factor = .99 , 
                    weight_entropy = 0.001, 
                    action_std_init = .5):

    # Create file to store model weigths
    save_f_path = setup_model_saving(algorithm = "REINFORCE")

    # Initialize buffers to store data for plotting
    plot_data = {'reward_history': [],
                 'timesteps':[]
                 }
    
    # Instantiate policy    
    policy_net = PolicyNetwork(input_size=env.observation_space.shape[0], 
                            output_size=env.action_space.shape[0],
                            )
        
    # Start timer
    start_time = time.time()
    # -----------------------------------------------------------------------------------
    # PLEASE DO NOT MODIFY THE CODE ABOVE THIS LINE
    # -----------------------------------------------------------------------------------        
    
    # Initialize variables
    counter_timesteps = 0
    best_reward = -np.inf

    # Instantiate value network
    value_net = ValueNetwork(input_size=env.observation_space.shape[0])

    # Instantiate optimisers
    optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=lr_policy_net)
    optimizer_value = torch.optim.Adam(value_net.parameters(), lr=lr_value_net)

    for episode in tqdm(range(int(max_episodes)), desc="Episode loop"):
       
        # Initialize buffer to record information about current episode
        trajectory = {}
        trajectory["values"] = []
        trajectory["actions"] = []
        trajectory["logprobs"] = []
        trajectory["rewards"] = []
        trajectory["entropies"] = []

        # Reset environment
        env.reset()
        state = env.state
        done = False

        # Run an episode and collect experience
        while not done:
            
            action , action_logprob , entropy = choose_action(state, policy_net, action_std_init)
            value = value_net(torch.from_numpy(state).float())
            
            next_state , reward , done , _ = env.step(action.detach().numpy().flatten())

            trajectory["values"].append(value)
            trajectory["logprobs"].append(action_logprob)
            trajectory["rewards"].append(reward)
            trajectory["entropies"].append(entropy)

            state = next_state
            counter_timesteps += 1
        
        logprobs = torch.stack(trajectory["logprobs"]).squeeze() # shape : (episode_length, )
        entropies = torch.stack(trajectory["entropies"]).squeeze() # shape : (episode_length, )
        values = torch.stack(trajectory["values"]).squeeze() # shape : (episode_length, )

        # Calculate discounted return at every time step
        discounted_return = 0
        returns = np.zeros_like(trajectory["rewards"], dtype=np.float32)
        for i in reversed(range(len(trajectory["rewards"]))):
            discounted_return = trajectory["rewards"][i] + discount_factor * discounted_return
            returns[i] = discounted_return
        returns = torch.tensor(returns , dtype=torch.float32)

        # Compute policy loss
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        loss_policy = (-1) * torch.mean(advantages.detach() * logprobs) + weight_entropy * ((-1) * torch.mean(entropies))

        if episode > 150:
            f=0
        # Compute value loss
        loss_value = torch.nn.functional.mse_loss(values , returns)

        # Update policy network
        optimizer_policy.zero_grad()
        loss_policy.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
        optimizer_policy.step()

        # Update Value Network
        optimizer_value.zero_grad()
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), float('inf'))
        optimizer_value.step()

        # Log evolution of the total return
        total_return = round(np.mean(sum(trajectory["rewards"])), 4)
        plot_data['reward_history'].append(total_return)
        plot_data['timesteps'].append(counter_timesteps)

        # Save best policy
        if total_return > best_reward:
            best_reward = total_return
            torch.save(policy_net.state_dict(), save_f_path)

        # -----------------------------------------------------------------------------------
        # PLEASE DO NOT MODIFY THE CODE BELOW THIS LINE
        # -----------------------------------------------------------------------------------
        # Check time
        if (time.time()-start_time) > max_time:
            print("Timeout reached: the best policy found so far will be returned.")
            break

    print(f"Policy model weights saved in: {save_f_path}") 
    print(f"Best reward: {best_reward}")

    return save_f_path , plot_data

#################################
# Helper functions
#################################
class ValueNetwork(torch.nn.Module):
    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def choose_action(state, policy_net, action_std):
    '''
    Sample action in continuous action space modelled with a Multivariate Normal distribution
    '''
    # Predict action mean from Policy Network
    action_mean = policy_net(torch.from_numpy(state).float())

    # Estimate action variance (decaying action std)
    action_var = torch.full(size=(policy_net.fc3.out_features,) , fill_value = action_std**2)
    cov_mat = torch.diag(action_var).unsqueeze(dim=0) 

    # Generate Multivariate Normal distribution with estimated mean and variance
    dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)

    # Sample action
    action = dist.sample()

    # Compute logprob and entropy
    logprob = dist.log_prob(action)
    entropy = dist.entropy()

    return action, logprob , entropy