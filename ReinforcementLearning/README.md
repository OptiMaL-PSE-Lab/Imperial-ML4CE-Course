# Reinforcement Learning for Inventory Management

## Overview

This coursework focuses on **inventory optimization using Reinforcement Learning (RL)**, in which you will have to code an optimization algorithm to identify the most effective order-placement policy.

## Task

Implement your RL algorithm in [`your_algorithm.py`](algorithms/your_algorithm.py). You have complete freedom in the choice of the optimization algorithm but, please, respect the template to ensure compliance with the evaluation platform we will use for grading the assignement.

Use the notebook [`ML4CE_RL_INV_CW.ipynb`](ML4CE_RL_INV_CW.ipynb) to execute and assess the performance of your algorithm. Learning curves and reward distribution plots are provided to analyse the training process and compare your algorithm against different benchmark policies, respectively.

### Grading of the assignement
- We will test the algorithm in `your_algorithm.py` on a different (potentially more challenging) environment by calling the function `your_optimization_alg(...)` multiple times with different seeds.
- The average of these runs will be the final score.
- **Limited evaluation budget** (per run): 
    - max_episodes = 2000  
    - max_time = 5 min

### Submission rules
- Use only the packages already provided in the python environment `ml4ce_rl.yml`. For more information about how to create an environment from an environment.yml file, visit [CONDA User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).
- Rename `your_alg.py` to `RL_your_team_name.py`
- Ensure your algorithm respects the template
    - Name of the function must be `your_optimization_alg(...)` 
    - Arguments passed to `your_optimization_alg(...)` 
    - Outputs of `your_optimization_alg(...)` should be a dict containing the weigths of the NN and another dict with plotting data.
- Do not modify the architecture of the neural network.
- Remember to modify the following lines at the end of [`your_algorithm.py`](algorithms/your_algorithm.py)
```python
    team_names = ["Del Rio Chanona, Antonio","Fons, Isabela"] # Names of the team members (Imperial format)
    cids = ["16879875", "06089513"] # CID (University Identifier)
    question = [1,0] # Would you like to be asked about this coursework in the final exam? 1: YES, 0: NO
```

### Recommendations
- Save policy parameters regularly during execution and and return the best value found when stopping criteria are met.
- Add parameters as keyword arguments with a default value.
- If you write any auxiliary function, make sure it is contained in the submitted `.py` file.

## Project Structure

```
ReinforcementLearning/
├── ML4CE_RL_INV_CW.ipynb               # Main notebook
├── README.md                           # This file
├── algorithms/                         # Algorithm implementations
│   ├── __init__.py
│   ├── reinforce.py                    # REINFORCE with baseline
│   ├── simulated_annealing.py          # Simulated Annealing (SA) algorithm
│   ├── heuristic_policy.py             # Heuristic (s,S) policy
│   └── your_algorithm.py               # Your algorithm template
├── benchmarking/                       # Auxiliary files for performance evaluation
│   ├── policy_REINFORCE_with_baseline.py     # Pretrained policy using REINFORCE with baseline
│   ├── policy_SA.py                    # Pretrained policy using SA
│   └── test_demand_dataset.pickle      # Test dataset
├── ML4CE_RL_environment.py             # RL environment
├── common.py                           # Auxiliary functions
├── utils.py                            # Plotting and data management functions
├── requirements.txt                    # Python environment
└── SCstructure.png                     # Environment diagram
```

## RL Environment
This project focuses on the three-echelon supply chain depicted below.
<p align="center">
  <img src=".\SCstructure.png" alt="SupplyChainStructure" width="500"/>
</p>

### Description
The **supplier** is an olive oil producer company, whose bottles are sold in different stores around the city. Due to the distance between the production facilities and the stores, the company owns a **distribution centre (DC)** in the vicinity of the city. **Retailers** sell the product directly to the customers and place replenishment orders to maintain sufficient stock levels. Likewise, the DC must keep enough inventory level to supply the stores and places replenishment orders directly to the manufacturing company. The challenge is to develop a re-order policy for each participant, since each stage faces uncertain in the demand of the stage succeeding it.

### Assumptions
- Customer demand is modeled as a random variable following a Poisson distribution.
- Production facilities have immediate access to an unlimited supply of raw materials.

### What happens during an episode?
Considering a time horizon of 4 weeks, at each day or time step $t$:
1.  DC and retailers place replenishment orders.
2.  DC and retailers receive orders after the corresponding lead time from their respective suppliers and update both inventory on-hand and pipeline inventory.
3.  Each stage satisfies demand of their respective clients according to current inventory levels. 
    1.  Backlogged sales take priority over the orders arriving at current period $t$.
    2.  Then, the orders placed by the retailers at the current period are fulfilled with the remaining available inventory.
    3.  Finally, the backlog of each retailer is updated.
4.  Profit is evaluated as the difference between the sales revenue and the different costs across the entire supply chain (i.e., delivery fees, variable order costs, holding cost, unfulfilled demand penalties and excess capacity cost).

### Elements of the Markov Decision Process (MDP)
* **Action space:** the agent must decide the number of units each retailer or DC reorders at each time step.
* **State space:** states represent the inventory position at each time step, which is the difference between the total inventory and the backlog.
* **Reward:** the agent tries to maximize the profit of the supply chain.
