# Williams-Otto Benchmark Coursework

This subfolder contains the coursework materials for the **Williams-Otto (WO) constrained optimization problem**. The Williams and Otto continuous stirred tank reactor (CSTR) is a widely studied example, frequently used to benchmark algorithms. This process, depicted below, involves feeding the reactor with two pure component streams, Fa and Fb (consisting of components A and B, respectively). Components A and B react to form an intermediate product, C, which further reacts with another B molecule to yield the desired products, P and E. A side reaction occurs between components C and P, resulting in the formation of a byproduct, G, which has no commercial value and is considered waste.

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
