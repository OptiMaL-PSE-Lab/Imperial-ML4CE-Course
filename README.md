# Machine Learning for Chemical Engineering (ML4CE)

Welcome to the **Machine Learning for Chemical Engineering (ML4CE)** course repository.  
This module introduces machine learning concepts for chemical engineering applications through coding projects.

ML4CE is a module offered at **Imperial College London** for 4th Year MEng and MSc students in the **Chemical Engineering Department**.  
The course introduces students to core machine learning concepts and their applications in solving chemical engineering problems.  
Assessment is based on mini-projects that apply machine learning and optimisation techniques to chemical engineering case studies.

This repository contains the material for the coursework coding projects.

---

## 📂 Coursework Structure

Currently, this repository has three parts:  

### 1. Unconstrained Coursework – PID Tuning

In this project, you will design a **data-driven optimisation algorithm** to tune a PID controller for a continuous stirred tank reactor (CSTR).  
The task is to optimise the controller gains such that the reactor system error is minimised.  

The optimisation problem involves balancing three aspects:
- Overall tracking error of the system  
- Magnitude of control actions (linked to operational cost)  
- Smoothness of control action changes (to avoid aggressive actuator behaviour)  

Your algorithm should find optimal PID parameters that yield stable, efficient control of the CSTR system.  

📖 [See README here](cw1_ddo/part1_unconstrained/README.md)

---

### 2. Constrained Coursework – Williams-Otto Problem

This project focuses on **constrained optimisation** of the **Williams-Otto (WO) chemical process**, a classic benchmark in chemical engineering.  
The objective is to maximise the process profit by adjusting two decision variables:  

- Reactor temperature (T<sub>R</sub>)  
- Flow rate of reactant B (F<sub>b</sub>)  

The optimisation must respect process constraints, such as limiting the use of reactant A and minimising waste generation.  
Each algorithm is evaluated under a strict evaluation budget, making efficient use of limited simulations critical.  

This coursework highlights the role of optimisation in process systems engineering, demonstrating how data-driven methods can be applied to complex, nonlinear chemical processes.  

📖 [See README here](cw1_ddo/part2_constrained/README.md)
📖 [See detailed problem description here](cw1_ddo/part2_constrained/ML4CE_WO_coursework.docx)

### 3. Data-driven MPC 

In this coursework, you will design a data-driven model predictive controller (MPC) to control a multistage extraction column. You have complete freedom to create your own data-gathering routine, data-driven model (neural network, decision tree etc.) and control horizon optimisation routine which all work together to form the controller. Since the takeaway is on optimising chemical engineering systems, we operate under the following assumption:
-	Evaluations are expensive, meaning that the runtime of the model training is limited by a fixed evaluation budget rather than a time budget constraint. This means that your exploration routine can only evaluate the system 5 times.
-	The data-gathering, model training and the control horizon optimisation routine will also be time limited to a budget of 2 mins for data-gathering and model training, and 5 min for controller optimisation (time to complete one simulation with the controller active)

Your team’s submission will be **three** functions:
  1. Data-gathering routine
  2. Model training
  3. Controller

These form your best attempt at controlling the multistage extraction column system using a data-driven MPC.

📖 [See README here](cw2_ddmpc/README.md)
📖 [See detailed problem description here](cw2_ddmpc/coursework_part_3.docx)


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
