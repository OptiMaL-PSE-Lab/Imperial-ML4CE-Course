import copy
import time

import numpy as np
import torch
import tqdm
from tqdm import tqdm

from common import PolicyNetwork
from ML4CE_RL_environment import MESCEnv
from utils import setup_model_saving


def your_optimization_alg(
    env: MESCEnv,
    policy_net: PolicyNetwork,
    *,
    max_episodes=2000,
    max_time=5 * 60,  # seconds
):
    # Create file to store model weigths
    save_f_path = setup_model_saving(algorithm="Your algorithm")

    # Initialize buffers to store data for plotting
    plot_data = {"reward_history": [], "episodes": []}

    start_time = time.time()
    best_reward = -np.inf
    best_policy = policy_net.state_dict()
    # -----------------------------------------------------------------------------------
    # PLEASE DO NOT MODIFY THE CODE ABOVE THIS LINE
    # -----------------------------------------------------------------------------------

    for episode in tqdm.tqdm(range(int(max_episodes))):

        # Your implementation goes here

        # -----------------------------------------------------------------------------------
        # PLEASE DO NOT MODIFY THE CODE BELOW THIS LINE
        # -----------------------------------------------------------------------------------
        # Check time
        if (time.time() - start_time) > max_time:
            print("Timeout reached: the best policy found so far will be returned.")
            break

    print(f"Policy model weights saved in: {save_f_path}")
    print(f"Best reward: {best_reward}")

    # Names of the team members (Imperial format)
    team_names = [
        "Del Rio Chanona, Antonio",
        "Fons, Isabela",
    ]
    # CID (University Identifier)
    cids = ["16879875", "06069513"]
    # Would you like to be asked about this coursework in the final exam? 1 if YES, 0 else
    question = [1, 0]

    return best_policy, plot_data
