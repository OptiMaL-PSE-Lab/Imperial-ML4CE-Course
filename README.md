# Machine Learning for Chemical Engineering (ML4CE)

Welcome to the **Machine Learning for Chemical Engineering (ML4CE)** course repository. This repository contains the material for present and past coursework coding assignments.

ML4CE is a module offered at **Imperial College London** for 4th Year MEng and MSc students in the **Chemical Engineering Department**. The course introduces students to core machine learning concepts and their applications in solving chemical engineering problems.  Assessment is based on mini-projects that apply machine learning and optimisation techniques to chemical engineering case studies.

**NOTE**: Please refer to the respective coursework's `README.md` for guidance on setting up your environment for the project. 

# 📂 Repo Structure

## 1. Data-Driven Optimization (DDO)

### Part 1: Unconstrained Optimization - PID Tuning

In this project, you will design a **data-driven optimisation algorithm** to tune a PID controller for a Continuous Stirred-Tank Reactor (CSTR). The optimization involves a 32-dimensional space (PID controller gains) with the objective to minimize control error.

Key features:
- Freedom to implement any DDO algorithm: direct, model-based, evolutionary search, etc.
- Limited by fixed evaluation budget (100 function evaluations)
- Challenge: balance overall tracking error, control action magnitude, and action smoothness

📖 [See README here](/DataDrivenOptimization/part1_unconstrained/README.md)  
📖 [Main benchmarking notebook](/DataDrivenOptimization/part1_unconstrained/ML4CE_CSTR_PID_CW.ipynb)  
📄 [PDF handout](/DataDrivenOptimization/part1_unconstrained/Coursework%20DDO%20Part%202.pdf)

### Part 2: Constrained Optimization - Williams-Otto Problem

This project focuses on **constrained optimisation** of the classic **Williams-Otto (WO)** chemical process benchmark. The objective is to maximize process profit by adjusting two decision variables while respecting process constraints.

Key features:
- Optimize reactor temperature (T<sub>R</sub>) and reactant B flow rate (F<sub>b</sub>)
- Work within constraints on reactant A usage and waste generation
- Limited evaluation budget makes efficient simulation use critical

📖 [See README here](/DataDrivenOptimization/part2_constrained/README.md)  
📖 [Benchmarking notebook](/DataDrivenOptimization/part2_constrained/ML4CE_WO_eval_algs.ipynb)  
📖 [Problem description](/DataDrivenOptimization/part2_constrained/ML4CE_WO_coursework.docx)  
📄 [PDF handout](/DataDrivenOptimization/part2_constrained/ML4CE_WO_coursework.pdf)

## 2. Data-Driven Model Predictive Control (DDMPC)

Design a data-driven model predictive controller to control a multistage extraction column. This coursework emphasizes the challenges of limited data and computational resources in real chemical engineering applications.

Key features:
- Create custom data-gathering routine, data-driven model, and control horizon optimization
- Limited to just 5 system evaluations for exploration
- Time-limited budgets: 2 minutes for data-gathering/model training, 5 minutes for controller simulation
- Submission requires three functions: data-gathering, model training, and controller

📖 [See README here](/DataDrivenMPC/readme.md)  
📖 [Problem description](/DataDrivenMPC/coursework_part_3.docx)  
📖 [Quick Start Notebook](/DataDrivenMPC/pc-gym/Quick_Start.ipynb)

## 3. Reinforcement Learning for Inventory Management

This coursework focuses on optimizing inventory management across a three-echelon supply chain using Reinforcement Learning (RL). The challenge is to develop efficient order-placement policies for retailers and distribution centers facing uncertain demand.

Key features:
- Complete freedom to implement any RL algorithm within the provided template
- Supply chain includes supplier, distribution center, and retailers
- Goal: Maximize profit while balancing inventory costs, backlog penalties, and delivery fees
- Benchmark against REINFORCE with baseline and simulated annealing algorithms

📖 [See README here](/ReinforcementLearning/README.md)  
📖 [Main notebook](/ReinforcementLearning/ML4CE_RL_INV_CW.ipynb)  
📖 [Environment specification](/ReinforcementLearning/ML4CE_RL_environment.py)


---
