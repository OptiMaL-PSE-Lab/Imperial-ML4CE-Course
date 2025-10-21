from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

##############################
# Example Exploration Scheme #
##############################
####################################
# Please modify the function below #
####################################
def explorer(x_t: np.array, u_bounds: dict, timestep: int) -> np.array:
    '''
    Function to collect more data to train the model.
    x_t (np.array) - Current state 
    u_bounds (dict) - Bounds on control inputs
    timestep (int) - Current timestep
    
    Output:
    u_plus - Next control input
    '''

    u_lower = u_bounds['low']
    u_upper = u_bounds['high']

    
    u_plus = np.random.uniform(u_lower, u_upper, size=u_lower.shape)
    
    return u_plus
    

########################
#Example Model Training#
########################
####################################
# Please modify the function below #
####################################
def model_trainer(data: np.array, env: callable) -> callable:
    """
    Trains a linear regression model using the provided data and environment parameters.
    Parameters:
    data (np.array): A tuple containing two numpy arrays: data_states and data_controls.
                    data_states is a 3D array with shape (reps, states, n_steps).
                    data_controls is a 3D array with shape (reps, controls, n_steps).
    env (callable): An environment object that contains the environment parameters.
    Returns:
    model (LinearRegression): The trained linear regression model.
    """
    data_states, data_controls = data
    
    # Select only states with indices 1 and 8
    selected_states = data_states[:, [1, 8], :]
    
    # Normalize the selected states and controls
    o_low, o_high = env.env_params['o_space']['low'][[1, 8]], env.env_params['o_space']['high'][[1, 8]]
    a_low, a_high = env.env_params['a_space']['low'], env.env_params['a_space']['high']
    selected_states_norm = (selected_states - o_low.reshape(1, -1, 1)) / (o_high.reshape(1, -1, 1) - o_low.reshape(1, -1, 1)) * 2 - 1
    data_controls_norm = (data_controls - a_low.reshape(1, -1, 1)) / (a_high.reshape(1, -1, 1) - a_low.reshape(1, -1, 1)) * 2 - 1

    # Get the dimensions
    reps, states, n_steps = selected_states_norm.shape
    _, controls, _ = data_controls_norm.shape
    
    # Prepare the data
    X_states = selected_states_norm[:, :, :-1].reshape(-1, states)
    X_controls = data_controls_norm[:, :, :-1].reshape(-1, controls)
    X = np.hstack([X_states, X_controls])
    y = selected_states_norm[:, :, 1:].reshape(-1, states)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LinearRegression(fit_intercept=False)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.3f}")

    return model




####################
#Example Controller#
####################
####################################
# Please modify the function below #
####################################
def controller(x: np.array, f: callable, sp: callable, env: callable, u_prev: np.array,) -> np.array:
    # Add names of team members and their respective CIDs
    controller.team_names = ['Max Bloor', 'Antonio Del Rio Chanona']
    controller.cids = ['01234567', '01234567']

    o_space = env.env_params['o_space']
    a_space = env.env_params['a_space']
    
    Q = 10 # state cost
    R = 500 # control cost
    
    horizon = 2 # Control Horizon
    x_current = x[1] # Current state

    n_controls = a_space['low'].shape[0]
    u_prev = (u_prev - a_space['low']) / (a_space['high'] - a_space['low']) * 2 - 1
    
    # Prediction function with data-driven model
    def predict_next_state(current_state, control):
        current_state_norm = (current_state - o_space['low'][[1, 8]]) / (o_space['high'][[1, 8]] - o_space['low'][[1, 8]]) * 2 - 1
        x = np.hstack([current_state_norm, control])
        prediction = f.predict(x.reshape(1, -1)).flatten()
        return (prediction + 1) / 2 * (o_space['high'][[1, 8]] - o_space['low'][[1, 8]]) + o_space['low'][[1, 8]]
    
    # Controller objective function 
    def objective(u_sequence):
        cost = 0
        x_pred = x_current
        for i in range(horizon):
            error = x_pred - sp
            cost += np.sum(error**2) * Q
            u_current = u_sequence[i*n_controls:(i+1)*n_controls]
            cost += np.sum((u_current - u_prev)**2) * R
            x_pred = predict_next_state(x_pred, u_current)
        return cost
    
    # Initial control guess and bounds (working in normalised control inputs)
    u_init = np.ones((horizon, 2)) * u_prev
    bounds = [(-1, 1)] * (horizon * n_controls)
    
    # Use scipy minimize to optimise the control cost
    result = minimize(objective, u_init.flatten(), method='powell', bounds=bounds)

    # Return the control input in the actual bounds
    optimal_control = result.x[:2]
    return (optimal_control + 1) / 2 * (a_space['high'] - a_space['low']) + a_space['low']
