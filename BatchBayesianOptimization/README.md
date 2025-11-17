# ML for Chemical Engineering - Batch Bayesian Optimisation Coursework 2025

This repository contains the material for the coursework coding project. 

---

## 📂 Coursework Task: Optimisation of a Bioprocess with Batch Bayesian Optimisation

This coursework involves the optimisation of a simulated bioprocess at process scale utilising CHO cells to produce a desired protein. Experimentally, this would involve a resource-intensive screening campaign involving the growth and feeding of cells under precise conditions (temperature, pH, feed amount, cell type, etc.) to maximize the production of a desired product. This coursework offers a simulated method of mapping bioprocess input parameters to a final predicted titre concentration: a measure of cell productivity. Your goal is to create a batch Bayesian Optimisation algorithm to obtain input parameters which maximizes titre concentration of the simulated bioprocess. 

---

## ❗ IMPORTANT UPDATES

1. Update to Training Point/Iteration/Batch budget constraint
- Instead of a maximum ceiling, we now impose a strict constant budget of 6 Training points, 15 iterations and 5 per batch constraint. The change is prompted by the change in marking criteria - see point number 2. 

2. Update to Marking Criteria
- Instead of evaluating all evaluations, only evaluations from iteration 3 to 15 will be summed and compared against the cohort. Since the training points, iteration and batch budget is now fixed to 6, 15 and 5 respectively, only elements from self.Y indexed 17 and onwards (iteration 3 to 15) will be summed and compared against the cohort.

3. Students are expected to code your own optimisation algorithm.
- You are *not* allowed to use pre-built algorithms or external packages that directly evaluate or optimise the functions.

---

## ❓ FAQs

1. Am I allowed to change the execution block?
- Yes! You are not only allowed to, but are encourage to! You should explore how best to build the search space and how to best choose your initial training points! (randomly? regular-sampling? uniform/non-uniform?)
- Make sure that your search space is within the bounds of each investigated variable. 
  
2. How many data points am I allowed for initial data points?
- Maximum of 6, there is no minimum. This does not count towards the max 15 iteration - 5 per batch limit imposed for your BO run. 

3. Can I use pandas?
- pandas is imported for the virtual lab. You should *not* include this as a part of your imports for your final submission.

4. Can I run analysis of the search space prior to choosing my training points?
- You are allowed to use any sampling method including random/Sobol, LHS etc. so long as the evaluation of these points are done within the BO class (such as the time spent evaluating them is captured)
- What you are *not* allowed to do for initial training point selection is any form of statistical analysis on the search space to choose your data points. 

## Getting Started
1. **Clone the repo:**
   ```bash
   git clone https://github.com/OptiMaL-PSE-Lab/Imperial-ML4CE-Course
   cd BatchBayesianOptimization
   ```

2. **Install Dependencies**
    
    conda (recommended):
    ```bash
    conda create --name myenv python=3.10 
    conda activate myenv
    pip install -r requirements.txt
    ```

    pip:
    ```bash
    pip install -r requirements.txt
    ```
