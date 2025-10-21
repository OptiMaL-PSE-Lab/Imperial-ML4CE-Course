# DDO Coursework Part 1: CSTR PID Controller Tuning

## Problem Definition

In this coursework, you will design a data-driven optimisation (DDO) algorithm to tune a PID controller embedded within a Continuous Stirred-Tank Reactor (CSTR) problem. You have complete freedom in what type of DDO algorithm you want to implement - direct, model-based, evolutionary search, etc.

### Key Constraints & Assumptions

- **Evaluation Budget**: Algorithms are limited by a fixed evaluation budget rather than time
- **Time Limit**: Maximum 1 minute execution time for 50 evaluations
- **Recommendation**: Have your algorithm exit and return the best value found if timing approaches the 1-minute mark
- **Objective**: Design a single DDO function that optimally tunes PID parameters to minimize error in the CSTR system

## Project Structure

```
part1_unconstrained/
├── ML4CE_CSTR_PID_CW.ipynb    # Main benchmarking notebook
├── README.md                   # This file
├── algorithms/                 # Algorithm implementations
│   ├── __init__.py
│   ├── COBYQA.py              # COBYQA algorithm
│   ├── Scipy_opt_algs.py      # SciPy optimization algorithms
│   ├── TuRBO.py               # TuRBO algorithm
│   └── your_alg.py            # Your algorithm template
├── bm_routine/                 # Benchmarking framework
│   ├── __init__.py
│   ├── CSTR_PID.py            # CSTR model and PID controller
│   ├── test_function.py       # Test function backend
│   └── utils.py               # Utility functions
└── images/                     # Generated plots and results
```

## Materials Provided

### Core Files

1. **`ML4CE_CSTR_PID_CW.ipynb`**
   - Main notebook for assessing algorithm performance
   - Compare your algorithm against state-of-the-art DDO methods
   - Generate trajectory plots and performance tables

2. **`CSTR_PID.py`**
   - Contains the CSTR model implementation
   - Includes untuned PID controller
   - Provides objective function for optimization
   - Features `opt_PID()` function for visualization

3. **`your_alg.py`**
   - Template algorithm implementation
   - Shows required function signature and return format
   - Currently implements Powell method with random search initialization
   - **Rename to `CW1_your_team_name.py` for submission**

4. **`test_function.py` & `utils.py`**
   - Backend benchmarking procedures (do not edit)
   - Already imported in the framework

## Getting Started

### 1. Understanding the Problem

The CSTR system has:
- **32-dimensional optimization space** (PID controller gains)
- **Objective**: Minimize control error
- **Challenge**: Expensive function evaluations

### 2. Algorithm Development

Modify `your_alg.py` with your DDO algorithm:

```python
def your_alg(f, x_dim, bounds, iter_tot):
    '''
    Your algorithm implementation
    
    Parameters:
    -----------
    f : test function object
        Contains f.fun_test(x) method for evaluation
    x_dim : int
        Problem dimensionality (32 for CSTR PID)
    bounds : array
        Variable bounds [lower, upper] for each dimension
    iter_tot : int
        Total evaluation budget
        
    Returns:
    --------
    x_best : array
        Best solution found
    f_best : float
        Best objective value
    team_names : list
        Your team member names
    cids : list
        Your student IDs
    '''
    # Your algorithm implementation here
    return x_best, f_best, team_names, cids
```

### 3. Testing Your Algorithm

Use the main notebook to benchmark your algorithm:

```python
# Add your algorithm to the test list
algorithms_test = [
    opt_COBYLA,
    your_alg,          # Your algorithm
    COBYQA,
    # TuRBO_Optimizer  # Uncomment to compare with TuRBO
]

# Run benchmarking
info, trajectories, timestamp = ML4CE_uncon_eval(
    N_x_l, f_eval_l, functions_test, algorithms_test, reps, home_dir
)
```

### 4. Performance Visualization

The framework provides:
- **Trajectory plots**: Show convergence over evaluations
- **Performance tables**: Statistical comparison across algorithms
- **PID response plots**: Visualize controller performance

```python
# Extract optimized gains and visualize CSTR response
K_opt = trajectories['D32']['cstr_pid_f']['x_list']['your_alg'][0][-1]
CSTR = CSTRSimulation()
CSTR.plot_result_ct(K_opt)
```

## Benchmarking Parameters

- **Dimensions**: 32 (fixed)
- **Function evaluations**: 100 per run
- **Repetitions**: 3 runs per algorithm
- **Starting points**: 15 different initializations
- **Test function**: `cstr_pid_f` (CSTR PID tuning)

## Algorithm Suggestions

Consider implementing:
- **Evolutionary algorithms** (GA, DE, PSO)
- **Bayesian optimization** (GP-based methods)
- **Model-based approaches** (surrogate models)
- **Hybrid methods** (combining multiple strategies)
- **Budget-aware strategies** (early termination, adaptive sampling)

## Submission Requirements

1. **Rename** `your_alg.py` to `CW1_your_team_name.py`
2. **Ensure** your algorithm follows the required function signature
3. **Test** thoroughly using the benchmarking notebook
4. **Document** your approach and results

## Performance Notes

The provided example algorithms (COBYQA, TuRBO) are state-of-the-art methods. Don't be discouraged if your algorithm doesn't outperform them immediately - focus on understanding DDO principles and implementing a robust solution.

## Tips for Success

1. **Start simple**: Begin with basic algorithms and gradually add sophistication
2. **Use the budget wisely**: Consider exploration vs. exploitation trade-offs
3. **Leverage problem structure**: PID tuning has known characteristics
4. **Monitor timing**: Ensure your algorithm respects the 1-minute limit
5. **Test extensively**: Use the benchmarking framework to validate performance

---

**Good luck with your DDO algorithm development!**