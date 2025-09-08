# Machine Learning for Chemical Engineering (ML4CE)

Welcome to the **Machine Learning for Chemical Engineering (ML4CE)** course repository.  
This module introduces machine learning concepts for chemical engineering applications through coding projects.

ML4CE is a module offered at **Imperial College London** for 4th Year MEng and MSc students in the **Chemical Engineering Department**.  
The course introduces students to core machine learning concepts and their applications in solving chemical engineering problems.  
Assessment is based on mini-projects that apply machine learning and optimisation techniques to chemical engineering case studies.

This repository contains the material for the coursework coding projects.

---

## 📂 Coursework Structure

The coursework is split into two parts:  

### 1. Unconstrained Coursework – PID Tuning (Part 2)

In this project, you will design a **data-driven optimisation algorithm** to tune a PID controller for a continuous stirred tank reactor (CSTR).  
The task is to optimise the controller gains such that the reactor system error is minimised.  

The optimisation problem involves balancing three aspects:
- Overall tracking error of the system  
- Magnitude of control actions (linked to operational cost)  
- Smoothness of control action changes (to avoid aggressive actuator behaviour)  

Your algorithm should find optimal PID parameters that yield stable, efficient control of the CSTR system.  

📖 [See README here](cw1_ddo/part1_unconstrained/README.md)

---

### 2. Constrained Coursework – Williams-Otto Problem (Part 3)

This project focuses on **constrained optimisation** of the **Williams-Otto (WO) chemical process**, a classic benchmark in chemical engineering.  
The objective is to maximise the process profit by adjusting two decision variables:  

- Reactor temperature (T<sub>R</sub>)  
- Flow rate of reactant B (F<sub>b</sub>)  

The optimisation must respect process constraints, such as limiting the use of reactant A and minimising waste generation.  
Each algorithm is evaluated under a strict evaluation budget, making efficient use of limited simulations critical.  

This coursework highlights the role of optimisation in process systems engineering, demonstrating how data-driven methods can be applied to complex, nonlinear chemical processes.  

📖 [See README here](cw1_ddo/part2_constrained/README.md)

---

## Getting Started
1. **Clone the repo:**
   ```bash
   git clone https://github.com/OptiMaL-PSE-Lab/Imperial-ML4CE-Course.git
   cd Imperial-ML4CE-Course
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
