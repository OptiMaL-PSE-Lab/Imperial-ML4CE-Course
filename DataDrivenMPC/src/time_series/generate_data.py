import numpy as np
###################################################
# You do not need to change anything in this file #
###################################################
def explorer(x_t: np.array, u_bounds: dict, timestep: int) -> np.array:
    '''
    Function to collect more data to train state-space model using step changes every 30 timesteps
    Inputs:
    x_t (np.array) - Current state 
    u_bounds (dict) - Bounds on control inputs
    timestep (int) - Current timestep
    
    Output:
    u_plus - Next control input
    '''
    u_lower = u_bounds['low']
    u_upper = u_bounds['high']
    step_duration = 30

    # Initialize function attributes if they don't exist
    if not hasattr(explorer, "current_input"):
        explorer.current_input = None
        explorer.steps_since_change = 0

    # Generate new input if it's the first call or time to change
    if explorer.current_input is None or explorer.steps_since_change >= step_duration:
        explorer.current_input = np.random.uniform(u_lower, u_upper, size=u_lower.shape)
        explorer.steps_since_change = 0

    explorer.steps_since_change += 1
    return explorer.current_input