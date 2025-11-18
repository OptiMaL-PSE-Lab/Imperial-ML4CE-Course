import copy
import time

import numpy as np
import torch
from tqdm import tqdm

from common import PolicyNetwork
from ML4CE_RL_environment import MESCEnv
from utils import setup_model_saving


def simulated_annealing_alg(
    env: MESCEnv,
    policy_net: PolicyNetwork,
    *,
    max_episodes=2000,
    max_time=5 * 60,  # seconds
    param_min=-1.0,
    param_max=1.0,
    num_episodes_avg=10,
    NNparams_0=None,
    initial_temp=1e5,
):

    # Define path to store best policies
    save_path = setup_model_saving(algorithm="SA")

    # Initialize buffers to store data for plotting
    plot_data = {
        "reward_history": [],
        "std_history": [],
        "best_reward_history": [],
        "episodes": [],
    }

    # Start timer
    start_time = time.time()
    # -----------------------------------------------------------------------------------
    # PLEASE DO NOT MODIFY THE CODE ABOVE THIS LINE
    # -----------------------------------------------------------------------------------

    # # INITIALIZATION
    # Parameters
    current_param = policy_net.state_dict() if NNparams_0 is None else NNparams_0
    best_param = copy.deepcopy(current_param)
    # Rewards
    current_reward, std = evaluate_avg_return(
        policy_net, env, num_episodes=num_episodes_avg
    )
    best_reward = copy.deepcopy(current_reward)
    counter_episodes = num_episodes_avg
    plot_data["episodes"].append(counter_episodes)
    plot_data["reward_history"].append(best_reward)
    plot_data["std_history"].append(std)
    plot_data["best_reward_history"].append(best_reward)

    # OPTIMIZATION LOOP
    max_iter = int((max_episodes - num_episodes_avg) / num_episodes_avg)
    for i in tqdm(range(max_iter), desc="Iteration loop"):

        # Sample a new policy from randomly
        candidate_param = sample_params(current_param, param_min, param_max)

        # Evaluate the candidate policy
        policy_net.load_state_dict(candidate_param)
        candidate_reward, std = evaluate_avg_return(
            policy_net, env, num_episodes=num_episodes_avg
        )

        # Check if the candidate policy is better than the current one
        if candidate_reward > best_reward:
            # Update the new best policy
            best_reward = candidate_reward
            best_param = copy.deepcopy(candidate_param)
            # Save policy
            torch.save(best_param, save_path)

        # Check if the candidate policy should be kept or discarded
        diff = candidate_reward - current_reward
        temp = initial_temp / (1 + i)  # update temperature paramter
        metropolis = np.exp(diff / temp)  # compute metropolis acceptance probability
        if diff > 0 or np.random.rand() < metropolis:
            # Update the current policy
            current_param = copy.deepcopy(candidate_param)
            current_reward = candidate_reward

        # Store the data for plotting
        counter_episodes += num_episodes_avg
        plot_data["episodes"].append(counter_episodes)
        plot_data["reward_history"].append(candidate_reward)
        plot_data["std_history"].append(std)
        plot_data["best_reward_history"].append(best_reward)

        # -----------------------------------------------------------------------------------
        # PLEASE DO NOT MODIFY THE CODE BELOW THIS LINE
        # -----------------------------------------------------------------------------------
        # Check execution time
        if (time.time() - start_time) > max_time:
            print("Timeout reached: the best policy found so far will be returned.")
            break

    print(f"Policy model weights saved in: {save_path}")
    print(f"Best reward found during training: {best_reward}")

    return best_param, plot_data


#################################
# Helper functions
#################################
def sample_params(params_prev, param_min, param_max):
    """
    Sample a random point in the neighborhood of a given point or value or the parameters (v). Tailored for EXPLOITATION purposes

    Explanation:
    sign = (torch.randint(2, (v.shape)) * 2 - 1) # This returns either -1 or 1
    eps = torch.rand(v.shape) * (param_max - param_min) # This returns the width of the step to be taken in the modification of the parameters
    Hence, the total update is: v + sign*eps.
    """
    params = {
        k: torch.rand(v.shape)
        * (param_max - param_min)
        * (torch.randint(2, (v.shape)) * 2 - 1)
        + v
        for k, v in params_prev.items()
    }
    return params


def evaluate_avg_return(policy_net, env, num_episodes=10, demand=None):
    """
    Runs a series of episodes and computes the average total return.

    Arguments:
    - policy_net --> Neural network that predicts the optimal action given the state
    - env --> Instance of MESCEnv environment
    - num_episodes --> Number of runs or episodes to estimate the average return (optional, default: 10)
    - demand --> List of scenario sets, containing realizations of customers' demand for each time step of each episode (optional, default: None)

    Returns:
    - mean_reward --> Average reward across runs
    - std_reward --> Standard deviation across runs
    """
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
            action_mean = policy_net(torch.FloatTensor(state))
            # TODO: Add covariance matrix and sample action from MultivariateNormal distribution
            action = np.fix(action_mean.detach().numpy())

            # Interact with the environment to get reward and next state
            state, reward, episode_terminated, _ = env.step(action)
            total_reward += reward

        reward_list.append(total_reward)

    # Compute mean and standard deviation
    mean_reward = np.mean(reward_list)
    std_reward = np.std(reward_list)

    return mean_reward, std_reward
