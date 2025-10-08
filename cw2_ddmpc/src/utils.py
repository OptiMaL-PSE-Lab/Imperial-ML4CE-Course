import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
###################################################
# You do not need to change anything in this file #
###################################################
def reward_fn(self, x:np.array, u:np.array, con: bool) -> float:
    Sp_i = 0
    cost = 0 
    R = 4
    for k in self.env_params["SP"]:
        i = self.model.info()["states"].index(k)
        SP = self.SP[k]

        o_space_low = self.env_params["o_space"]["low"][i] 
        o_space_high = self.env_params["o_space"]["high"][i] 

        x_normalized = (x[i] - o_space_low) / (o_space_high - o_space_low)
        setpoint_normalized = (SP - o_space_low) / (o_space_high - o_space_low)

        r_scale = self.env_params.get("r_scale", {})

        cost += (np.sum(x_normalized - setpoint_normalized[self.t]) ** 2) * r_scale.get(k, 1)

        Sp_i += 1
    u_normalized = (u - self.env_params["a_space"]["low"]) / (
        self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"]
    )

    # Add the control cost
    cost += R * u_normalized**2
    r = -cost
    return r

def rollout(env:Callable,  explore:bool, explorer=None, controller = None, model=None) -> tuple[np.array, np.array]:
    nsteps = env.N
    x_real_shape = env.x0.shape[0] - len(env.SP)
    x_log = np.empty((1,env.x0.shape[0] - len(env.SP), env.N))
    u_log = np.empty((1,env.Nu, env.N))
    u_prev = env.env_params['a_space']['high']-(env.env_params['a_space']['high']-env.env_params['a_space']['low'])
    x, _ = env.reset()
    for i in range(nsteps):
        sp = get_sp_i(env,i)
        if explore:
            u = explorer(x, env.env_params['a_space'], i)
        else:
            u = controller(x[:x_real_shape], model, sp, env, u_prev)
            u_prev = u
        x, _, _, _, _  = env.step(u)
        x_log[0, :, i] = x[:x_real_shape]
        u_log[0, :, i] = u
    
    return x_log, u_log

def get_sp_i(env, i):
    sp = env.SP
    sp_values = []
    for k in sp:
        sp_array = sp[k]
        sp_values.append(sp_array[i])
    return np.array(sp_values)

def plot_simulation_results(x_logs, u_logs, env,):
    """
    Plot specific states with setpoints and all controls. If multiple logs are provided, plot median with shaded areas for max and min.
    
    Args:
    x_logs (np.array or list of np.array): State log(s) for single or multiple simulations
    u_logs (np.array or list of np.array): Control log(s) for single or multiple simulations
    env: Environment object containing information about the system

    """
    state_indices=[1]

    N_reps = x_logs.shape[2]
    N_steps = env.N
    N_states = len(state_indices)

    sp_array = np.array([env.SP['Y1']])
    # Calculate median, min, and max for states and controls
    x_median = np.median(x_logs, axis=2) if N_reps > 1 else x_logs[0]
    x_min = np.min(x_logs, axis=2) if N_reps > 1 else x_logs[0]
    x_max = np.max(x_logs, axis=2) if N_reps > 1 else x_logs[0]

    u_median = np.median(u_logs, axis=2) if N_reps > 1 else u_logs[0]
    u_min = np.min(u_logs, axis=2) if N_reps > 1 else u_logs[0]
    u_max = np.max(u_logs, axis=2) if N_reps > 1 else u_logs[0]



    # Create time array
    time = np.linspace(0, env.tsim, N_steps)
    labels = ['Y1', 'L', 'G']
    colours = ['tab:red', 'tab:green', 'tab:blue']
    # Plot specified states
    fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharex=True)

    for i, state_idx in enumerate(state_indices):
        axs[i].plot(time, x_median[state_idx], label=labels[i] if N_reps == 1 else 'State (Median)', color = colours[i])
        if N_reps > 1:
            axs[i].fill_between(time, x_min[state_idx], x_max[state_idx], alpha=0.2, label='State (Min-Max Range', color = colours[i], edgecolor = 'None')
        axs[i].step(time, sp_array[i,:], color = 'black', linestyle = '--', label='Setpoint')
        axs[i].legend()
        axs[i].set_xlabel('Time [hr]')
        axs[i].set_xlim(0, env.tsim)
        axs[i].grid(True)
    
    for i in range(1, 3):
        axs[i].step(time, u_median[i-N_states], label=labels[i] if N_reps == 1 else f'{labels[i]} (Median)',  color = colours[i])
        if N_reps > 1:
            axs[i].fill_between(time, u_min[i-N_states], u_max[i-N_states], alpha=0.2, label=f'{labels[i]} (Min-Max Range)', color = colours[i], edgecolor = 'None', step = 'post')
        axs[i].set_ylabel(labels[i])
        axs[i].set_xlabel('Time [hr]')
        axs[i].set_xlim(0, env.tsim)
        axs[i].legend()
        axs[i].grid(True)
    
    axs[-1].set_xlabel('Time')
    plt.tight_layout()

    
    axs[i].legend()
    
    plt.tight_layout()
    plt.show()

def visualise_collected_data(data_states, data_controls, num_simulations, figsize=(10, 5)):
    """
    Visualize the collected simulation data.
    
    Parameters:
    - data_states: numpy array of shape (N_sim, num_states, time_steps)
    - data_controls: numpy array of shape (N_sim, num_controls, time_steps)
    - num_simulations: number of simulations to plot (default: 5)
    - figsize: size of the figure (default: (15, 10))
    """
    num_controls = data_controls.shape[1]
    time_steps = data_states.shape[2]
    
    fig, axs = plt.subplots(1, 3, figsize=figsize, sharex=True)
    fig.suptitle('Collected and Provided data', fontsize=16)
    
    # Plot state variables
    ax = axs[0]
    for j in range(num_simulations+5):
        if j < 5:
            ax.plot(range(time_steps), data_states[j, 1, :], alpha=0.7, color = 'tab:red', label='Provided' if j == 0 else '')
        else:
            ax.plot(range(time_steps), data_states[j, 1, :], alpha=0.7, label= 'Collected' if j == 5 else '', color = 'tab:green')
    ax.set_ylabel('Y1')
    ax.grid(True)
    ax.legend(loc='upper right')
    
    # Plot control inputs
    for i in range(num_controls):
        ax = axs[i+1]
        for j in range(num_simulations+5):
            if j < 5:
                ax.step(range(time_steps), data_controls[j, i, :], alpha=0.7,color = 'tab:red', label='Provided' if j == 0 else "")
            else:
                ax.step(range(time_steps), data_controls[j, i, :], alpha=0.7,color = 'tab:green', label='Collected' if j == 5 else "")
        ax.set_ylabel('L' if i == 0 else 'G' )
        ax.grid(True)
        ax.legend(loc = 'upper right')
    
    axs[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.show()