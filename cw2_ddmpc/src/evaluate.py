import numpy as np
import time
from pcgym import make_env # Give it a star on Github :)
from utils import reward_fn, rollout, plot_simulation_results, visualise_collected_data # functions to make the code below more readable 
from your_alg import explorer, model_trainer, controller # Your algorithms (you may need to change the "your_alg" to the actual name)
###################################################
# You do not need to change anything in this file #
###################################################

#######################################
#  Multistage Extraction Column model #
#######################################
T = 100
nsteps = 60

action_space = {
  'low': np.array([5, 10]),
  'high':np.array([500, 1000])
}

observation_space = {
  'low' : np.array([0]*10+[0.3]+[0.3]),
  'high' : np.array([1]*10+[0.4]+[0.4])  
}
SP = {
        'Y1': [0.5 for i in range(int(nsteps/2))] + [0.7 for i in range(int(nsteps/2))]
    }
env_params_ms = {
  'N': nsteps,
  'tsim': T,
  'SP': SP,
  'o_space': observation_space,
  'a_space': action_space,
  'x0': np.array([0.55, 0.3, 0.45, 0.25, 0.4, 0.20, 0.35, 0.15, 0.25, 0.1,0.3,0.3]),
  'model': 'multistage_extraction', 
  'noise': True, #Add noise to the states
  'noise_percentage':0.01,
  'integration_method': 'casadi',
  'custom_reward': reward_fn,
  'normalise_o': False,
  'normalise_a': False,
}


#########################
#Import pre-defined data#
#########################

env = make_env(env_params_ms)
N_sim = int(5) # Budget for simulations to gather extra data (DO NOT CHANGE THIS)
data_states = np.load('./time_series/data_states.npy')
data_controls = np.load('./time_series/data_control.npy')


###################
#Exploration Phase#
###################
reps = 5
current_scores = []
for rep in range(reps):
  print(f"\nRepetition {rep + 1}/{reps}")  
  start_time = time.time()
  for i in range(N_sim):
    x_log, u_log = rollout(env=env, explore=True, explorer=explorer)
    data_states = np.append(data_states, x_log, axis=0)
    data_controls = np.append(data_controls, u_log, axis=0)
  end_time = time.time()
  exploration_time = end_time - start_time

  print(f'Exploration time: {exploration_time:.2f} seconds')

  visualise_collected_data(data_states, data_controls, num_simulations=N_sim)
  data = (data_states, data_controls)

  #################
  # Training Phase#
  #################
  start_time = time.time()
  model = model_trainer(data,env)
  end_time = time.time()
  model_training_time = end_time - start_time
  print(f'Model training time: {model_training_time:.2f} seconds')


  ###############
  #Control Phase#
  ###############

  N_reps = 3 # Repetitions of simulation with data-driven controller
  num_setpoints = 3

  setpoints = [0.2, 0.3, 0.4, 0.5, 0.6]

  scores = []
  execution_times = []

  for setpoint_index in range(num_setpoints):
    SP = {
      'Y1': [setpoints[setpoint_index] for i in range(int(nsteps/2))] + [setpoints[setpoint_index+2] for i in range(int(nsteps/2))]
    }
    
    env_params_ms['SP'] = SP
    env = make_env(env_params_ms)
    
    x_log = np.zeros((env.x0.shape[0] - len(env.SP), env.N, N_reps))
    u_log = np.zeros((env.Nu, env.N, N_reps))
    for i in range(N_reps):
      start_time = time.time()
      x_log[:, :, i], u_log[:, :, i] = rollout(env=env, explore=False, controller=controller, model=model)
      end_time = time.time()

      execution_time = end_time - start_time
      execution_times.append(execution_time)
      print(f'Controller Execution time: {execution_time:.2f} seconds')
      print(f'Total (inc. exploration and training): {exploration_time + model_training_time + execution_time:.2f} seconds')

    plot_simulation_results(x_log, u_log, env)
    score = np.sum((np.median(x_log[1,:,:], axis = 1) - env.SP['Y1']))**2 + 0.00001*np.sum((u_log)**2)
    print("Score for Setpoint", setpoint_index + 1, ":", score)
    scores.append(score)

    average_execution_time = np.mean(execution_times) 
    print(f'Average total time: {average_execution_time + model_training_time + exploration_time:.2f} seconds')

    total_normalized_score = np.sum(scores)/num_setpoints
    
    average_execution_time = np.mean(execution_times)
  current_scores.append(total_normalized_score)

df = {'team': controller.team_names, 'CIDs': controller.cids, 'score': min(current_scores)}
print(df)