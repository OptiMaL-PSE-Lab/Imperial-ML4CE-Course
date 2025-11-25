# ML for Chemical Engineering - Batch Bayesian Optimisation Coursework 2025

This repository contains the material for the coursework coding project. 

---

## 📂 Coursework Task: Optimisation of a Bioprocess with Batch Bayesian Optimisation

This coursework involves the optimisation of a simulated bioprocess at process scale utilising CHO cells to produce a desired protein. Experimentally, this would involve a resource-intensive screening campaign involving the growth and feeding of cells under precise conditions (temperature, pH, feed amount, cell type, etc.) to maximize the production of a desired product. This coursework offers a simulated method of mapping bioprocess input parameters to a final predicted titre concentration: a measure of cell productivity. Your goal is to create a batch Bayesian Optimisation algorithm to obtain input parameters which maximizes titre concentration of the simulated bioprocess. 

---

## ❗ IMPORTANT UPDATES

### A. Updated Marking Procedure for Batch Bayesian Optimization Coursework
We have now updated the marking criteria. The marking will be based on the best function value you have discovered so far, starting from Batch 3 onwards.
How the marking works:

- Initialization + Batches 1 and 2 (first 16 evaluations):These do not contribute to your mark. Their purpose is purely for initialization and early exploration.
- Batch 3: After you submit the five evaluations from Batch 3, We will look at all evaluations so far (16 from earlier batches + 5 from Batch 3). Your score for Batch 3 will be the highest single function value among all these points.
- Batch 4: You submit five new samples. We evaluate them, and then consider all evaluations to date (16 + 5 + 5). Your score for Batch 4 will again be the best function value found so far, even if that point was found in an earlier batch.
- Batch 5-onwards: Same procedure: you submit five new points, they are evaluated, and we take the best value so far as your Batch score. At the end we add your scores for each batch, and this is your final score. If your algorithm stops before finishing all interactions due to time, we take your highest evaluation and use that for the missing batches.

##### What this means for you:
- Your algorithm can explore: exploratory samples will not penalize you, since the score is based on the best-so-far value.
- The marking is such that:
- Algorithms that find good points early will be rewarded.
- Algorithms that achieve high values in general will also be rewarded.
- Your score for each assessed batch is simply: the highest evaluation value your algorithm has achieved up to that point.

### B. Update to Training Point/Iteration/Batch budget constraint
- Instead of a maximum ceiling, we now impose a strict constant budget of 6 Training points, 15 iterations and 5 per batch constraint. The change is prompted by the change in marking criteria - see point A. 

### C. Students are expected to code your own BO optimisation algorithm.
- You are *not* allowed to use pre-built algorithms or external packages that directly evaluate or optimise the functions and you must code a BO optimizer/algorithm.
- The only instance where you are allowed scipy.optimize.minimize is for obtaining your hyperparameters via optimisationof the log marginal likelihood. 

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
