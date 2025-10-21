import gymnasium as gym
import numpy as np
from scipy.stats import poisson
import copy
import itertools

class MESCEnv():   
    def __init__(self, structure, *args, **kwargs):
        '''
        Args:
            structure : List of integers whose length is the number of stages and each element is the number of participants in that stage.
        '''
        self.name = "InventoryManagement"
        # Default values for the initial conditions (overriden if kwargs are passed)
        self.n_periods = kwargs.get("num_periods", 4*7) # 4 weeks * 7 days
        self.demand_dist_fcn = poisson
        self.demand_dist_param = [{'mu': 5}, # Mo-Fri
                                  {'mu': 7}] # Sat-Sun
        self.seed = 0

        # Buffers to store data
        self.demand_dataset = None # Attribute to store the demands of the episode if externally provided (e.g., during evaluation)

        # Assigned the sub-classed frozen distribution to the demand distribution function
        if self.demand_dist_fcn is poisson:
            self.demand_dist = frozen_poisson(random_state=self.seed)

        # Update initial values if kwargs are passed
        for key, value in kwargs.items():
            setattr(self, key, value)  

        # Definition of the supply chain
        self.structure = structure # Number of retailers and DCs in the system
        # Instantiate retailers        
        self.retailers = []
        for i in range(sum(structure[0])):
            self.retailers.append(Retailer())

        # Instantiate DCs  
        self.DCs = []
        for i in range(len(structure[1])):
            self.DCs.append(DC(self.retailers[(i*structure[0][i]):(i*structure[0][i]+structure[0][i])]))

        self.reset()

        # Define action space: reorder quantities of each stage m at each period t
        self.n_actions = len(self.retailers) + len(self.DCs) # Number of actions = number of retailers + number of DCs
        self.action_space_UB = np.array([retailer.order_quantity_limit for retailer in self.retailers] + [DC.order_quantity_limit for DC in self.DCs])
        self.action_space = gym.spaces.Box(low=np.zeros(self.n_actions, dtype=np.int16), 
                                           high=self.action_space_UB, 
                                           dtype=np.int16)

        # Define observation space: inventory position of each entity + day of the week
        self.n_states  = len(self.retailers) + len(self.DCs) + 1 # Number of states = number of retailers + number of DCs + day of the week 
        self.observation_space_UB = np.array([retailer.capacity + retailer.order_quantity_limit for retailer in self.retailers] + [DC.capacity + DC.order_quantity_limit for DC in self.DCs] + [6])
        self.observation_space = gym.spaces.Box(low=np.zeros(self.n_states, dtype=np.int16), 
                                                high=self.observation_space_UB, 
                                                dtype=np.int16)
    
    @property
    def seed(self):
        return self._seed
    @seed.setter
    def seed(self, value):
        self._seed = value
        if self.demand_dist_fcn is poisson:
            self.demand_dist = frozen_poisson(random_state=self.seed)

    def _RESET(self):
        self.current_period = 1
        self.dow = 0
        
        # Definition of the self.demands_episode attribute, which stores the demand information that will be actually used during self.step()
        if self.demand_dataset is None:
            self.demands_episode , self.prob_per_scenario = self.sample_demands_episode()
        else:
            self.demands_episode = copy.deepcopy(self.demand_dataset)

        for i, retailer in enumerate(self.retailers):
            retailer.reset(self.demands_episode[:,i])
        for DC in self.DCs:
            DC.reset()
        
        self._update_state()
    def reset(self):
        return self._RESET()

    def _STEP(self, action):
        #> 0. Ensure action is of the correct dtype
        action = action.astype(np.int16) 

        for i in range(len(self.DCs)):
            #> 1.1 DCs place orders to supplier
            # TODO: revise it's being done correctly. Depending on the meaning, it should take one action or the other.
            self.DCs[i].place_order(action[i+len(self.retailers)], self.current_period)

            #> 1.2 DCs receive orders from supplier
            self.DCs[i].receive_order(self.current_period)

            #> 1.3 DCs satisfy demand from retailers
            self.DCs[i].satisfy_demand(self.DCs[i].retailers, action[:len(self.retailers)], self.current_period)
        
        for i in range(len(self.retailers)):
            #> 2.1 Retailers receive orders from DCs
            self.retailers[i].receive_order(self.current_period)

            #> 2.2 Retailers satisfy demand from customers
            self.retailers[i].satisfy_demand(self.current_period)
        
        #> 3. Compute reward
        revenue = np.sum(retailer.SD * retailer.unit_price for retailer in self.retailers) # Revenue from sale
        holding_cost = np.sum(DC.I * DC.holding_cost for DC in self.DCs) + np.sum(retailer.I * retailer.holding_cost for retailer in self.retailers) 
        fixed_order_cost = np.sum(DC.fix_order_cost*DC.n_orders for DC in self.DCs) + np.sum(retailer.fix_order_cost*retailer.n_orders for retailer in self.retailers)
        var_order_cost = np.sum(DC.var_order_cost * action[i+len(self.retailers)] for i, DC in enumerate(self.DCs))
        penalty_unsatisfied_demand = np.sum(retailer.UD * retailer.lost_sales_cost for retailer in self.retailers) 
        
        def saturating_penalty(x, coef, scale):
            '''
            Instead of scaling linearly forever, applies tanh function to preserve a strong penatly for small violations while avoiding unbounded spikes.
            
            Motivation:
                It was identified that capacity violation seldom happened. For this reason, the capacity violation coefficient was set large.
                However, some actions led to nearly unbounded penalties for capacity violation, becoming extreme points of the reward function that should be avoided.
                This observation has motivated the use of a saturating penalty function.
            
            Arguments:
            - x: surplus units
            - coef: saturation point of the penalty (i.e., max penalty to be applied)
            - scale: how fast saturation appears. It can be seen as the slope of the tanh function before saturation. Smaller scale = steeper slope.
                     If scale = capacity/2, going half over the capacity gives ~70% of maximum penalty.
            '''
            return coef * np.tanh(x/scale)
        penalty_capacity_violation = np.sum(saturating_penalty(DC.I_surplus, DC.capacity * DC.capacity_violation_cost, DC.capacity*.75) for DC in self.DCs) \
                                   + np.sum(saturating_penalty(retailer.I_surplus, retailer.capacity * retailer.capacity_violation_cost, retailer.capacity*.75)  for retailer in self.retailers) # Penalty for exceeding storage capacity
        
        reward = revenue - holding_cost - fixed_order_cost - var_order_cost - penalty_unsatisfied_demand - penalty_capacity_violation
        # print(f"Period {self.current_period} - Reward: {reward}, Revenue: {revenue}, Penalty Capacity Violation: {penalty_capacity_violation}, SurplusDC: {np.sum(DC.I_surplus for DC in self.DCs)}, SurplusR: {np.sum(r.I_surplus for r in self.retailers)}, Holding Cost: {holding_cost}, Fixed Order Cost: {fixed_order_cost}, Variable Order Cost: {var_order_cost}, Penalty Unsatisfied Demand: {penalty_unsatisfied_demand}")
        
        #> 4. Update period
        self.current_period += 1

        #> 5. Update state
        self._update_state()

        #> 6. Check if episode is finished
        if self.current_period > self.n_periods:
            episode_ended = True
        else:
            episode_ended = False

        return self.state , reward, episode_ended , {}
    def step(self, action):
        return self._STEP(action)
    
    def _update_state(self):
        self.dow = (self.current_period-1) % 7 # Day of the week (0-6)
        self.state = np.array([retailer.inv_pos for retailer in self.retailers] + [DC.inv_pos for DC in self.DCs] + [self.dow])

    def sample_demands_episode(self):
        '''
        Sample demand for each retailer in the current period. The demand distribution function is defined in the __init__ method.
        '''
        demands_episode = []
        prob_per_scenario = np.zeros(self.n_periods, dtype=np.float32)
        for i in range(self.n_periods):
            if self.dow < 5:
                demand_dist_param = self.demand_dist_param[0]
            else:
                demand_dist_param = self.demand_dist_param[1]

            scenario = self.demand_dist.rvs(**demand_dist_param, size=len(self.retailers))
            prob_per_scenario[i] = np.prod([self.demand_dist.pmf(scenario[j], **demand_dist_param) for j in range(len(scenario))])
            demands_episode.append(scenario)

        return np.array(demands_episode), prob_per_scenario 
    
class DiscreteMESCEnv():   
    def __init__(self, structure = [[2],[1],1], *args, **kwargs):
        '''
        Args:s
            structure : List of integers whose length is the number of stages and each element is the number of participants in that stage.
        '''
        self.name = "DiscreteInventoryManagement"
        # Default values for the initial conditions (overriden if kwargs are passed)
        self.n_periods = kwargs.get("num_periods", 4*7) # 4 weeks * 7 days
        self.demand_dist_fcn = poisson
        self.demand_dist_param = {'mu': 10}
        self.seed = 0

        # Buffers to store data
        self.demand_dataset = None # Attribute to store the demands of the episode if externally provided (e.g., during evaluation)

        # Assigned the sub-classed frozen distribution to the demand distribution function
        if self.demand_dist_fcn is poisson:
            self.demand_dist = frozen_poisson(random_state=self.seed)

        # Update initial values if kwargs are passed
        for key, value in kwargs.items():
            setattr(self, key, value)  

        # Definition of the supply chain
        self.structure = structure # Number of retailers and DCs in the system
        # Instantiate retailers        
        self.retailers = []
        for i in range(sum(structure[0])):
            self.retailers.append(Retailer())

        # Instantiate DCs  
        self.DCs = []
        for i in range(len(structure[1])):
            self.DCs.append(DC(self.retailers[(i*structure[0][i]):(i*structure[0][i]+structure[0][i])]))

        # Define action space: reorder quantities of each stage m at each period t
        self.action_mapping = [[0, 10, 20] for _ in range(len(self.retailers))] + [[0, 50, 100] for _ in range(len(self.DCs))]
        self.action_space = gym.spaces.MultiDiscrete(nvec = [len(m) for m in self.action_mapping],
                                                     dtype=np.int16,
                                                     seed=None, start=None)
        self.action_space.n_indep = self.action_space.nvec.sum() # number of actions, when considered independent
        self.action_space.n = self.action_space.nvec.prod() # number of joint actions
        self.action_lookup = {tuple(action): i for i, action in enumerate(list(itertools.product(*self.action_mapping)))}
        
        # Define observation space: inventory position of each entity + day of the week
        # CAUTION: IN THE ORIGINAL ENVIRONMENT, THE OBSERVATION SPACE DEFINES DCS FIRST, BUT HERE IT WAS CHANGED TO ALIGN WITH ACTIONS
        self.observation_mapping = [[0, 1/3 , 2/3 , 1] * ret.capacity for ret in self.retailers] + [[0, 1/3, 2/3, 1] * DC.capacity for DC in self.DCs]
        self.observation_space = gym.spaces.MultiDiscrete(nvec = [len(m) for m in self.observation_mapping] ,
                                                                  dtype=np.int16,
                                                                  seed=None, start=None) # TODO: check why is it unused (also in original env IMP_CW_env.py)
        self.observation_space.n = self.observation_space.nvec.prod() # number of joint possible states
        self.observation_lookup = {tuple(obs): i for i,obs in enumerate(list(itertools.product(*self.observation_mapping)))}

        self.reset()
    
    @property
    def seed(self):
        return self._seed
    @seed.setter
    def seed(self, value):
        self._seed = value
        if self.demand_dist_fcn is poisson:
            self.demand_dist = frozen_poisson(random_state=self.seed)
        
    def _RESET(self):
        self.current_period = 1        
        # Definition of the self.demands_episode attribute, which stores the demand information that will be actually used during self.step()
        if self.demand_dataset is None:
            self.demands_episode , self.prob_per_scenario = self.sample_demands_episode()
        else:
            self.demands_episode = copy.deepcopy(self.demand_dataset)

        for i, retailer in enumerate(self.retailers):
            retailer.reset(self.demands_episode[:,i])
        for DC in self.DCs:
            DC.reset()
        
        self._update_state()
        self._discretize_state(self.state)
    def reset(self):
        return self._RESET()

    def _STEP(self, action):
        #> 0. Ensure action has the correct type
        action = np.astype(action, np.int16)

        for i in range(len(self.DCs)):
            #> 1.1 DCs place orders to supplier
            # TODO: revise it's being done correctly. Depending on the meaning, it should take one action or the other.
            self.DCs[i].place_order(action[i+len(self.retailers)], self.current_period)

            #> 1.2 DCs receive orders from supplier
            self.DCs[i].receive_order(self.current_period)

            #> 1.3 DCs satisfy demand from retailers
            self.DCs[i].satisfy_demand(self.DCs[i].retailers, action[:len(self.retailers)], self.current_period)
        
        for i in range(len(self.retailers)):
            #> 2.1 Retailers receive orders from DCs
            self.retailers[i].receive_order(self.current_period)

            #> 2.2 Retailers satisfy demand from customers
            self.retailers[i].satisfy_demand(self.current_period)
        
        #> 3. Compute reward
        revenue = np.sum(retailer.SD * retailer.unit_price for retailer in self.retailers) # Revenue from sale
        holding_cost = np.sum(DC.I * DC.holding_cost for DC in self.DCs) + np.sum(retailer.I * retailer.holding_cost for retailer in self.retailers) 
        fixed_order_cost = np.sum(DC.fix_order_cost*DC.n_orders for DC in self.DCs) + np.sum(retailer.fix_order_cost*retailer.n_orders for retailer in self.retailers)
        var_order_cost = np.sum(DC.var_order_cost * action[i+len(self.retailers)] for i, DC in enumerate(self.DCs))
        penalty_unsatisfied_demand = np.sum(retailer.UD * retailer.lost_sales_cost for retailer in self.retailers) 
        def saturating_penalty(x, coef, scale):
            '''
            Instead of scaling linearly forever, applies tanh function to preserve a strong penatly for small violations while avoiding unbounded spikes.
            
            Motivation:
                It was identified that capacity violation seldom happened. For this reason, the capacity violation coefficient was set large.
                However, some actions led to nearly unbounded penalties for capacity violation, becoming extreme points of the reward function that should be avoided.
                This observation has motivated the use of a saturating penalty function.
            
            Arguments:
            - x: surplus units
            - coef: saturation point of the penalty (i.e., max penalty to be applied)
            - scale: how fast saturation appears. It can be seen as the slope of the tanh function before saturation. Smaller scale = steeper slope.
                     If scale = capacity/2, going half over the capacity gives ~70% of maximum penalty.
            '''
            return coef * np.tanh(x/scale)
        penalty_capacity_violation = np.sum(saturating_penalty(DC.I_surplus, DC.capacity * DC.capacity_violation_cost, DC.capacity*.75) for DC in self.DCs) \
                                   + np.sum(saturating_penalty(retailer.I_surplus, retailer.capacity * retailer.capacity_violation_cost, retailer.capacity*.75)  for retailer in self.retailers) # Penalty for exceeding storage capacity
        reward = revenue - holding_cost - fixed_order_cost - var_order_cost - penalty_unsatisfied_demand - penalty_capacity_violation
        # print(f"Period {self.current_period} - Reward: {reward:.1f}, Revenue: {revenue}, Penalty capacity: {penalty_capacity_violation:.1f}, Holding Cost: {holding_cost}, Fixed Order Cost: {fixed_order_cost}, Variable Order Cost: {var_order_cost}, Penalty Unsatisfied Demand: {penalty_unsatisfied_demand}")
        #> 4. Update period
        self.current_period += 1

        #> 5. Update state
        self._update_state() # This function updates the "true state" of the environment in self.state

        #> 6. Check if episode is finished
        if self.current_period > self.n_periods:
            episode_ended = True
        else:
            episode_ended = False

        #> 7. Map environment state (true values) to values that agent can observe (fixed in self.observation_mapping)
        self._discretize_state(self.state)

        return self.obs , reward, episode_ended , {}
    def step(self, action):
        return self._STEP(action)
    
    def _update_state(self):
        self.state = np.array([retailer.inv_pos for retailer in self.retailers]+ [DC.inv_pos for DC in self.DCs])

    def _discretize_state(self, state):
        '''
        POMDP: the environment knows the "true state" (i.e., the actual inventory position), but the agent only observes
        a discretized version of it, since the observation_space is only allowed to take 4 discrete values.
        '''
        self.obs = np.array([self._probabilistic_rounding(value, bins) for value, bins in zip(state, self.observation_mapping)], dtype=np.int16) 
    
    def _probabilistic_rounding(self, value, bins):
        if value <= min(bins):
            return min(bins)
        elif value >max(bins):
            return max(bins)
        else: # find neighbors
            for i in range(len(bins)):
                if bins[i] <= value <= bins[i+1]:
                    low, high = bins[i], bins[i+1]
                    p_low = 1. - (value - low) / (high - low)
                    p_high = 1. - p_low
                    return np.random.choice([low, high], p=[p_low,p_high])
            
    def sample_demands_episode(self):
        '''
        Sample demand for each retailer in the current period. The demand distribution function is defined in the __init__ method.
        '''
        demands_episode = []
        prob_per_scenario = np.zeros(self.n_periods, dtype=np.float32)
        for i in range(self.n_periods):
            demand_dist_param = self.demand_dist_param
            scenario = self.demand_dist.rvs(**demand_dist_param, size=len(self.retailers))
            prob_per_scenario[i] = np.prod([self.demand_dist.pmf(scenario[j], **demand_dist_param) for j in range(len(scenario))])
            demands_episode.append(scenario)

        return np.array(demands_episode), prob_per_scenario

    def sample_action(self, probabilities:tuple=None):
        '''
        NOTE: feature of passing probabilities is not used because it would imply that actions are independent and it has been decided to work with joint actions. 
        - Reason: probability argument taken by the sample method of the MultiDiscrete class should be a tuple of ndarrays, each array with the probabilities of each one of the values that each action can take.
        - Reference: https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.MultiDiscrete.sample
        # TODO: check if that assumption is ok
        '''
        action_idx = self.action_space.sample(probability=probabilities)
        return np.array([values[idx] for idx,values in zip(action_idx, self.action_mapping)], dtype=np.int16)

class DC():
    def __init__(self, retailers , *args, **kwargs):
        # Defaul values for the initial conditions (overriden if kwargs are passed)
        self.I0 = 100 # Initial on-hand inventory
        self.lead_time = 2
        self.capacity = 150 # storage capacity of the DC
        self.order_quantity_limit = 200 # productive capacity of DC's supplier. TBD: check how now it's unused
        self.holding_cost = .5
        self.fix_order_cost = 50
        self.var_order_cost = 2 # It was set in the parent class in the original article
        self.capacity_violation_cost = 1 # Penalty of exceeding storage capacity expressed in cost/item

        self.retailers = retailers # List of retailers served by the DC
        
        self.reset()
        
    def reset(self):
        # Use the correct dtype
        self.order_quantity_limit = np.array(self.order_quantity_limit, dtype=np.int16)
        self.capacity = np.array(self.capacity, dtype=np.int16)
        # Initialize variables
        self.I = np.array(self.I0, dtype = np.int16) # On-hand inventory at the start of each period
        self.I_surplus = np.array(0, dtype=np.int16) # Number of units by which capacity constraint is violated at the start of each period
        self.T = np.array(0, dtype=np.int16) # Pipeline inventory at the start of each period
        self.backlog = np.array(0, dtype=np.int16) # Quantity that DC requests to supplier but supplier denies/is unable to provide
        self.inv_pos = self.I+self.T-self.backlog # Inventory position = On hand inventory + Pipeline inventory - Backlogged demand
        # Initialize action record
        self.action_log = [] # Orders placed to supplier
        # Initialize counters
        self.n_orders = 0 # Number of orders placed by the DC to the supplier in the current period

    def place_order(self, action , current_period):
        '''
        Args:
            action : ordered quantity from the DC to the supplier
            current period
        Returns: 
            accepted order: stored in action_log as a list [arrival_time , ordered_quantity]
            backlog : difference between actual and accepted order
        '''
        if action > 0 :
            self.action_log.append([current_period + self.lead_time , min(self.order_quantity_limit, action)])
            self.backlog = np.minimum(0 , action - self.order_quantity_limit, dtype=np.int16) # Backlog is the difference between the order placed and the order accepted
            self.n_orders = 1 # Reset the number of orders placed by the DC to the supplier in the current period
        else:
            self.n_orders = 0 # Reset the number of orders placed by the DC to the supplier in the current period

    def receive_order(self, current_period):
        # Receive order
        if len(self.action_log) > 0 :
            if current_period == self.action_log[0][0] :
                # self.I = min(self.capacity , self.I + self.action_log[0][1])
                self.I += self.action_log[0][1]
                self.action_log.pop(0)
        
        # Check if inventory at hand surpasses storage capacity
        self.I_surplus = max(0, self.I-self.capacity)

    def satisfy_demand(self, retailers, actions, current_period):

        #> 0. Initialize arrarys to accumulate demnand satisfaction to compute the number of orders placed
        total_demand_satisfied = np.zeros(len(retailers), dtype=np.int16) # sum of backlogged and current demand satisfied in current period

        #> 1. Prioritize satisfaction of backlogged demand
        if np.sum(retailer.backlog for retailer in retailers) > 0 :
            for i, retailer in enumerate(retailers):
                if self.I <= retailer.backlog :
                    retailer.backlog -= self.I
                    total_demand_satisfied[i] += self.I
                    self.I = np.array(0, dtype=np.int16)
                else:
                    self.I -= retailer.backlog
                    total_demand_satisfied[i] += retailer.backlog
                    retailer.backlog = np.array(0, dtype=np.int16)

        #> 2. Satisfy current period demand
        if self.I > 0 :
            for i, retailer in enumerate(retailers):
                if self.I <= actions[i]:
                    retailer.backlog = actions[i] - self.I
                    total_demand_satisfied[i] += self.I
                    self.I = np.array(0, dtype=np.int16)
                else:
                    self.I -= actions[i]
                    total_demand_satisfied[i] += actions[i]
                    retailer.backlog = np.array(0, dtype=np.int16)
        else:
            for i, retailer in enumerate(retailers):
                retailer.backlog = actions[i]

        #> 3. Assess the number of batches into which the order placed by each retailer must be divided
        for i, retailer in enumerate(retailers):
            quantity_left = total_demand_satisfied[i]
            n_orders = 0
            while quantity_left > 0:
                if quantity_left > retailer.order_quantity_limit:
                    retailer.order_arrival_list.append((current_period + retailer.lead_time , retailer.order_quantity_limit))
                    quantity_left -= retailer.order_quantity_limit
                else:
                    retailer.order_arrival_list.append((current_period + retailer.lead_time , quantity_left))
                    quantity_left = 0
                n_orders += 1
            retailer.n_orders = n_orders # Store the number of orders placed by each retailer to the DC in the current period
        
        #> 4. Update pipeline inventory and total inventory position
        self.T = np.sum(order[1] for order in self.action_log)
        self.inv_pos = self.I + self.T - np.sum(retailer.backlog for retailer in retailers) # Inventory position = On hand inventory + Pipeline inventory - Backlogged demand

class Retailer():
    def __init__(self, *args, **kwargs):
        self.I0 = 25
        self.lead_time = 1
        self.capacity = 75
        self.order_quantity_limit = 50
        self.holding_cost = 1
        self.fix_order_cost = 10
        self.unit_price = 50
        self.lost_sales_cost = 5
        self.capacity_violation_cost = 1 # Penalty for exceeding storage capacity expressed in cost/item
        
        self.reset()

    def reset(self, *args):
        # Use the correct dtype
        self.order_quantity_limit = np.array(self.order_quantity_limit, dtype=np.int16)
        self.capacity = np.array(self.capacity, dtype=np.int16)
        # Initialize variables
        self.I = np.array(self.I0, dtype=np.int16)
        self.I_surplus = np.array(0, dtype=np.int16) # Number of units by which capacity constraint is violated at the start of each period
        self.T = np.array(0, dtype=np.int16) # Pipeline inventory at the start of each period
        self.SD = np.array(0, dtype=np.int16) # Sales performed at each period 
        self.UD = np.array(0, dtype=np.int16) # Unsatisfied demand at each period
        self.backlog = np.array(0, dtype=np.int16) # Quantity that retailer requests to DC but DC denies/is unable to provide
        self.inv_pos = self.I + self.T - self.backlog # Inventory position = On hand inventory + Pipeline inventory - Backlogged demand
        self.order_arrival_list = [] # Orders placed to supplier
        # Initialize counters
        self.n_orders = 0 # Number of orders placed by the retailer to the supplier in the current period
        # Initialize action record
        if len(args) > 0:
            self.demands_episode = args[0] # Demand log for the current episode

    def place_order(self):
        '''
        orders to be sent from DC to retailer are placed and computed in the method "place_order" of the DC class
        '''

    def receive_order(self, current_period):
        n_orders = 0
        new_order_arrival_list = []

        for arrival_time, order_quantity in self.order_arrival_list:
            if arrival_time == current_period:
                self.I += order_quantity
                n_orders += 1
            else:
                new_order_arrival_list.append((arrival_time, order_quantity))
        self.order_arrival_list = new_order_arrival_list

        # Check if inventory at hand surpasses storage capacity
        self.I_surplus = max(0, self.I-self.capacity)

        return n_orders
    
    def satisfy_demand(self, current_period):
        demand = self.demands_episode[current_period-1] 
        self.SD = np.minimum(demand, self.I, dtype=self.I.dtype)
        self.I -= self.SD
        self.UD = demand - self.SD
        self.T = np.sum(order for _ , order in self.order_arrival_list)
        self.inv_pos = self.I + self.T - 0 # Inventory position = On hand inventory + Pipeline inventory. Retailers do not have backlogged demand; unfulfilled final customer demand is lost at a penalty cost.

class frozen_poisson():
    '''
    This class is used to freeze the parameters of the poisson distribution. It is used to avoid passing the parameters every time a new sample is generated.
    '''
    def __init__(self, random_state=None):
        self.random_state = random_state

    def rvs(self, mu = 0 , size=1):
        if not hasattr(self, 'rng'):
            if self.random_state is None:
                self.rng = np.random.Generator(np.random.PCG64())
            else:
                self.rng = np.random.Generator(np.random.PCG64(seed=self.random_state))
        return self.rng.poisson(mu, size=size)

    def pmf(self, value , mu = 0):
        return poisson.pmf(value, mu=mu)