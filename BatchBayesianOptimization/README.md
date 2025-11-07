# ML for Chemical Engineering - Batch Bayesian Optimisation Coursework 2025

This repository contains the material for the coursework coding project. 

---

## 📂 Coursework Task: Optimisation of a Bioprocess with Batch Bayesian Optimisation

This coursework involves the optimisation of a simulated bioprocess at process scale utilising CHO cells to produce a desired protein. Experimentally, this would involve a resource-intensive screening campaign involving the growth and feeding of cells under precise conditions (temperature, pH, feed amount, cell type, etc.) to maximize the production of a desired product. This coursework offers a simulated method of mapping bioprocess input parameters to a final predicted titre concentration: a measure of cell productivity. Your goal is to create a batch Bayesian Optimisation algorithm to obtain input parameters which maximizes titre concentration of the simulated bioprocess. 

---

## ❓ FAQs

1. Am I allowed to change the execution block?
- Yes! You are not only allowed to, but are encourage to! You should explore how best to build the search space and how to best choose your initial training points! (randomly? regular-sampling? uniform/non-uniform?)
- Make sure that your search space is within the bounds of each investigated variable. 
  
2. How many data points am I allowed for initial data points?
- Maximum of 6, there is no minimum.

3. Can I use pandas?
- pandas is imported for the virtual lab. You should *not* include this as a part of your imports for your final submission. 


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
