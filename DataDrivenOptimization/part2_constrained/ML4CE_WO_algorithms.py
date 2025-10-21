from scipy.optimize import minimize as min_la
from cobyqa import minimize as min_qa

###########################################################################################
####################################### EXAMPLES ##########################################
###########################################################################################

def COBYLA(f, bounds, budget, i_rep):
    constraints = [
        {"type": "ineq", "fun": f.WO_con1_test},
        {"type": "ineq", "fun": f.WO_con2_test},
    ]

    x_start = f.x0[i_rep].flatten()

    opt = min_la(
        f.fun_test,
        x_start,
        method="COBYLA",
        constraints=constraints,
        bounds=bounds,
        options={"maxiter": budget, "disp": False},
    )

    team_names = ["Mathias Neufang", "Antonio Del Rio Chanona"]
    cids = ["01234567", "01234567"]
    return team_names, cids


def COBYQA(f, bounds, budget, i_rep):
    constraints = [
        {"type": "ineq", "fun": f.WO_con1_test},
        {"type": "ineq", "fun": f.WO_con2_test},
    ]

    x_start = f.x0[i_rep].flatten()

    opt = min_qa(
        f.fun_test,
        x_start,
        bounds=bounds,
        constraints=constraints,
        options={"maxfev": budget, "disp": False},
    )

    team_names = ["Mathias Neufang", "Antonio del Rio Chanona"]
    cids = ["01234567", "01234567"]
    return team_names, cids
