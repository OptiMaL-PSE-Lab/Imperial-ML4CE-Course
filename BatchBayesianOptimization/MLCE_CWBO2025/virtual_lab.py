import numpy as np
from scipy.integrate import solve_ivp
import MLCE_CWBO2025.conditions_data as data
import pandas as pd


class EXPERIMENT:
    def __init__(
            self, 
            T: float = 32, 
            pH: float = 7.2, 
            cell_type: str = "celltype_1", 
            reactor: str = "3LBATCH", 
            feeding: list = [(10, 0), (20, 0), (30, 0)], 
            time=150
        ):

        """
        EXPERIMENT Class simulates CHO kinetics depending on the initialization:

        T: Temperature of the reactor in type float

        pH: pH of the reactor in type float

        cell_type: Used cell culture in the experiment (e.g. 'cell_culture1')

        reactor: Used reactor scale and type (e.g. '3LBATCH')

        feeding: Feeding regime as a list of time and concentration tuples 
        (e.g. [(50, 0), (80, 0), (100, 0), (120, 10)])
        """

        df = data.df
        params = df[(df['reactor'] == reactor) & (df['cell_type'] == cell_type)]

        # Initialize lists for five different cell cultures
        self.reactor    = reactor
        self.volume     = 3 #L
        self.cell_type  = cell_type
        self.time       = time #Experiment time
        self.my_max     = params["my_max"].iloc[0]                       # Maximum growth rate (1/h)
        
        self.K_lysis    = params["K_lysis"].iloc[0]

        self.K_L, self.K_A, self.K_G, self.K_Q = params["K"].iloc[0] # in mM
        self.Y = params["Y"].iloc[0] # in [ cells*mmol^(-1) / cells*mmol^(-1) / - / - ]
        self.m = params["m"].iloc[0] # in mmol * cell^(-1) * h^(-1)
        self.k_d_Q, self.k_d_max, self.k_my = params["k"].iloc[0]

        self.A      = params["A"].iloc[0] 
        self.E_a    = params["E_a"].iloc[0] 
        self.pH_opt = params["pH_opt"].iloc[0] 

        # P, X_T, X_V, X_D, G, Q, L, A = x
        self.initial_conditions = [0, 1e6, 0.8 * 1e6, 0, 210, 1, 9, 0]
        self.solution = None # concentration values
        self.t = None        # integration time values


        # Cell-specific inputs
        self.T         = T   # Temperature (Celsius)
        self.pH        = pH   # Current pH

        # Feeding plan
        self.feeding = feeding # List of (time, new_G_value)
        
        self.R = 8.314  # Universal gas constant (J/(mol·K))

    def temperature_effect(self):
        x = self.T
        mu = self.E_a
        A = 5

        left_part = np.exp(-1 * ((x - mu) / 10)**2)
        right_part = np.exp(-0.9 * ((x - mu) / 3.6)**2)  # Faster decay on the right

        factor = A * np.where(x < mu, left_part, right_part)
        return factor
    
    # Function to calculate pH effect
    def pH_effect(self) -> float:
        x = self.pH
        mu = self.pH_opt
        A = 2

        left_part = np.exp(-0.8 * ((x - mu) / 1)**2)
        right_part = np.exp(-1 * ((x - mu) / 0.5)**2)  # Faster decay on the right

        factor = A * np.where(x < mu, left_part, right_part)
        return factor

    def my(self, G, Q, L, A):
        '''
        Calculates the Growth Rate depending on:
            
            G - glucose concentration
            Q - glutamine concentration
            L - lactate concentration
            A - ammonia concentration
            
        Returns a growth rate value in h^(-1)
        '''
        temperature_factor = self.temperature_effect()
        pH_factor = self.pH_effect()

        my_max = self.my_max
        K_G = self.K_G
        K_Q = self.K_Q
        K_L = self.K_L
        K_A = self.K_A

        my = my_max * G/(K_G + G) * Q/(K_Q + Q) * K_L/(K_L + L) * K_A/(K_A + A) * temperature_factor * pH_factor
        return my
    

    # Set up system of ODEs
    def ODE(self,t,x):
        '''
        Calculate the rate of change for given initial conditions

            G - glucose concentration
            Q - glutamine concentration
            L - lactate concentration
            A - ammonia concentration
            X_T - total cell density
            X_V - viable cell density
            X_D - dead cell density

        Returns a list of gradients
        '''
        P, X_T, X_V, X_D, G, Q, L, A = x

        # Growth rate
        my = self.my(G, Q, L, A)

        # Cell death rate
        k_d = self.k_d_max * (self.k_my/(my + self.k_my))
        K_lysis = self.K_lysis
        k_d_Q = self.k_d_Q
        K_G = self.K_G
        
        # Yield rations
        Y_X_G, Y_X_Q, Y_L_G, Y_A_Q, Y_P_X, Y_dot_P_X = self.Y
        m_G, m_Q = self.m
        
        # Rate of change of cell densities
        dX_T_dt = my * X_V - K_lysis * X_D
        dX_V_dt = (my-k_d) * X_V
        dX_D_dt = k_d * X_V - K_lysis * X_D

        dP_dt = Y_P_X * X_T + Y_dot_P_X * (my * G / (K_G + G)) * X_V

        # Rate of change of substrate
        dG_dt = X_V * (-my/Y_X_G - m_G)
        dQ_dt = X_V * (-my/Y_X_Q - m_Q) - k_d_Q * Q
        dL_dt = -X_V * Y_L_G * (-my/Y_X_G - m_G)
        dA_dt = -X_V * Y_A_Q * (-my/Y_X_Q - m_Q) + k_d_Q * Q

        gradients = [dP_dt, dX_T_dt, dX_V_dt, dX_D_dt, dG_dt, dQ_dt, dL_dt, dA_dt]

        return gradients

    # Segmented solving of ODEs
    def ODE_solver(self):
        """
        Solves the ODE system with feeding events, updating G dynamically.
        """
        t_span = (0, self.time)  # Total time for the simulation
        t_eval_total = []  # Store all time points
        y_total = []       # Store all solutions
        current_t = 0      # Current time
        current_y = self.initial_conditions.copy()  # Initial conditions

        for event_time, new_G_value in self.feeding:
            # Solve until the next feeding event
            t_span_segment = (current_t, event_time)
            t_eval_segment = np.linspace(current_t, event_time, 1000)
            
            solution = solve_ivp(
                fun=self.ODE,
                t_span=t_span_segment,
                y0=current_y,
                t_eval=t_eval_segment,
                method="RK45"
            )
            
            # Append results of this segment
            t_eval_total.extend(solution.t)
            y_total.append(solution.y)
            
            # Update current time and state for the next segment
            current_t = event_time
            current_y = solution.y[:, -1]
            if current_y[4] < new_G_value:
                current_y[4] = new_G_value  # Update G based on feeding event

            new_Q_value = new_G_value * 0.4
            if current_y[5] < new_Q_value:
                current_y[5] = new_Q_value # Update Q based on feeding event

        # Final integration from the last feeding event to the end
        t_span_segment = (current_t, self.time)
        t_eval_segment = np.linspace(current_t, self.time, 500)
        
        solution = solve_ivp(
            fun=self.ODE,
            t_span=t_span_segment,
            y0=current_y,
            t_eval=t_eval_segment,
            method="RK45"
        )
        
        # Append final segment
        t_eval_total.extend(solution.t)
        y_total.append(solution.y)

        # Combine all segments into a single array
        t_eval_total = np.array(t_eval_total)
        y_total = np.hstack(y_total)
        
        self.solution = y_total
        self.solution[0] = self.solution[0] / (self.volume * 1e3) # Transform unit into g/L

        self.t = t_eval_total
        #print("Simulation Complete")
        return y_total
 
    def measurement(self, noise_level=None, quantity="P") -> tuple:
        """
        The OD value is proportional to quantity with added noise.
        
        Returns:
            A tuple containing the noisy measurements and their corresponding indices.
        """

        np.random.seed(1234)

        reactor_type = self.reactor
        if (noise_level == None) and (reactor_type in data.noise_level):
            noise_level = data.noise_level[reactor_type]
        elif noise_level != None:
            pass
        else:
            raise ValueError(f"Unknown reactor type: {reactor_type}. Please provide a valid reactor type or a custom noise level.")
        
        self.ODE_solver()
        
        index = {"P": 0, "X_T": 1, "X_V": 2, "X_D": 3, "G": 4, "Q": 5, "L": 6, "A": 7}

        true_value = self.solution[index[quantity]][-1]
        
        noise_magnitude = max(noise_level * true_value, 1e-8)
        noise = np.random.normal(0, noise_magnitude)
        noisy_value = true_value + noise
        
        return noisy_value

def conduct_experiment(X, initial_conditions: list = [0, 0.4 * 1e9, 0.4 * 1e6, 0, 20, 3.5, 0, 1.8], noise_level=None):
    """
        Returns objective function value given:

        X - Multidimensional input
        initial_conditions - Initial conditions of the experimental environment
        noise_level - Measurement uncertainty (None results in no noise)
    """

    result = []
    feeding = [(10, 0), (20, 0), (30, 0)] # feeding input for 2d experiments
    reactor = "3LBATCH"

    for row in X:
        if len(row) == 6: 
            T, pH, F1, F2, F3, cell_type = row
            fidelity = 0
            feeding = [(40, float(F1)), (80, float(F2)), (120, float(F3))] # feeding time and amount
            reactor = data.reactor_list[int(np.round(fidelity))]
        else:
            raise ValueError(f"Cannot handle the dimensionality of X. n must be 6 but is {len(row)}")

        cell = EXPERIMENT(T=T, pH=pH, time=150, feeding=feeding, reactor=reactor, cell_type=cell_type)
        cell.initial_conditions = initial_conditions
        value = float(cell.measurement(quantity="P", noise_level=noise_level))
        #print(value)
        result.append(value)
    return result
