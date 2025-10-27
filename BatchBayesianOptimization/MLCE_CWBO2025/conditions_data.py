import pandas as pd

reactor_list = [ "3LCONTBATCH"]

process_parameters = {

    "3LCONTBATCH": {
        "celltype_1": {"my_max": 0.035, "K_lysis": 4e-2,   "k": [1e-3, 1e-2, 1e-2],          "K": [150, 40, 1, 0.22],    "Y": [9.23e7, 8.8e8, 1.6, 0.68, 6.2292e-8, 4.41e-6],    "m": [8e-13, 3e-12], "A": 1e1, "pH_opt": 7.2, "E_a": 32},
        "celltype_2": {"my_max": 0.046, "K_lysis": 1.5e-2, "k": [1.1e-3, 1.05e-2, 1e-2],     "K": [155, 42, 1.1, 0.23],  "Y": [2.3e8, 1.6e9, 2.1, 0.95, 6.2292e-8, 4.41e-6],   "m": [1.15e-10, 5e-12], "A": 1.2e1, "pH_opt": 6.8, "E_a": 35},
        "celltype_3": {"my_max": 0.045, "K_lysis": 1.8e-2, "k": [1.05e-3, 1.04e-2, 1.02e-2], "K": [153, 41, 1.05, 0.21], "Y": [2.25e8, 1.55e9, 2.05, 0.92, 6.2292e-8, 4.41e-6],"m": [1.12e-10, 3e-12], "A": 1.1e1, "pH_opt": 6.5, "E_a": 34},}
}

noise_level = {   
            "3LCONTBATCH": 8e-2,
        }

fidelity_cost = {   
            "3LCONTBATCH": 0.5,
        }

data = []
for reactor, cell_data in process_parameters.items():
    for cell_type, params in cell_data.items():
        entry = {
            "reactor": reactor,
            "cell_type": cell_type,
            **params
        }
        data.append(entry)

df = pd.DataFrame(data)