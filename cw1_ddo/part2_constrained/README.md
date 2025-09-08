# Williams-Otto Benchmark Coursework

This subfolder contains the coursework materials for the **Williams-Otto (WO) constrained optimization problem**, used to test data-driven optimization algorithms.  
The main repo already introduces the problem—this README focuses on the provided code structure and usage, and the .docx gives details on the problem.

---

## 📂 Folder Structure

- **`ML4CE_MyAlg.py`**  
  Wrapper file for your algorithm.  
  ➝ *Insert your implementation here — this is also the file you will submit.*

- **`ML4CE_WO_eval_algs.ipynb`**  
  Jupyter notebook for benchmarking.  
  ➝ *Run this to evaluate and compare algorithms (including yours) and to generate plots.*

- **`ML4CE_WO.py`**  
  Core implementation of the Williams-Otto problem.  
  ➝ *Contains the objective function (profit maximization) with temperature and reactant B flowrate as decision variables.*  
  (*No modifications required.*)

- **`ML4CE_WO_Wrapper.py`**  
  Wrapper for the WO problem.  
  ➝ *No modifications required.*

- **`ML4CE_WO_algorithms.py`**  
  Example algorithms provided for testing and benchmarking.  
  ➝ *Your final submission is graded only against other student algorithms.*  
  (*No modifications required.*)

- **`ML4CE_WO_utils.py`**  
  Utility functions for benchmarking and plotting results.  
  ➝ *No modifications required.*

- **`ML4CE_WO_requirements.txt`**  
  Dependencies required for running the coursework.  
  ➝ *Install with:*  
  ```bash
  pip install -r ML4CE_WO_requirements.txt
