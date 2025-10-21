import numpy as np 
import torch
import time
import tqdm 

from common import PolicyNetwork
from utils import setup_model_saving

def your_optimization_alg(  env, *,  
                            max_episodes = 200 , 
                            max_time = 60. ,
                            ):

    # Create file to store model weigths
    save_f_path = setup_model_saving(algorithm = "Your algorithm")
    
    # Initialize buffers to store data for plotting
    plot_data = {'reward_history': [],
                 'timesteps':[]
                 }
    
    # Initialize policies    
    policy_net = PolicyNetwork(input_size=env.observation_space.shape[0], 
                                output_size=env.action_space.shape[0],
                               )
    
    start_time = time.time()
    best_reward = -np.inf
    # -----------------------------------------------------------------------------------
    # PLEASE DO NOT MODIFY THE CODE ABOVE THIS LINE
    # -----------------------------------------------------------------------------------  

    for episode in tqdm.tqdm(range(int(max_episodes))):

        # Your implementation goes here
        
        # -----------------------------------------------------------------------------------
        # PLEASE DO NOT MODIFY THE CODE BELOW THIS LINE
        # -----------------------------------------------------------------------------------
        # Check time
        if (time.time()-start_time) > max_time:
            print("Timeout reached: the best policy found so far will be returned.")
            break

    print(f"Policy model weights saved in: {save_f_path}") 
    print(f"Best reward: {best_reward}")

    team_names = ["Student1","Student2"]
    cids = ["1234", "5678"]
    question = [0,0] # 1 if RL 0 else

    return save_f_path , plot_data
