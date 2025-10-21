import numpy as np
from cobyqa import minimize
from scipy.optimize import direct, Bounds
import scipy


#########################
# --- Random search --- #
#########################

def Random_search(f, n_p, bounds_rs, iter_rs):
    '''
    This function is a naive optimization routine that randomly samples the 
    allowed space and returns the best value.

    n_p: dimensions
    iter_rs: number of points to create
    '''

    # arrays to store sampled points
    localx   = np.zeros((n_p,iter_rs))  # points sampled
    localval = np.zeros((iter_rs))        # function values sampled
    # bounds
    bounds_range = bounds_rs[:,1] - bounds_rs[:,0]
    bounds_bias  = bounds_rs[:,0]

    for sample_i in range(iter_rs):
        x_trial = np.random.uniform(0, 1, n_p)*bounds_range + bounds_bias # sampling
        localx[:,sample_i] = x_trial
        localval[sample_i] = f.fun_test(x_trial)
    # choosing the best
    minindex = np.argmin(localval)
    f_b      = localval[minindex]
    x_b      = localx[:,minindex]

    return f_b, x_b


#########################
# ----- Algorithm ----- #
#########################

def COBYQA(
        f,
        x_dim, #TODO here the functionality of multiple input dimensions has to be implemented
        bounds,
        f_eval_, # length of trajectory (objective function evaluation budget)
        has_x0 = False,
        ): 
    '''
    params: parameters that define the rbf model
    X:      matrix of previous datapoints
    take_x_start: boolean to either receive the starting point from f or to randomly sample it
    in case the starting point is randomly sampled, the number of samples is deducted from the evaluation budget. Mind, that 
    the number of samples 
    in case the starting point is taken from the function, this is treated as us providing the algorithm with a starting point.
    Therefore this counts as a single evaluation from the evaluation budget
    '''

    if has_x0 == True:
        iter_ = f_eval_ - 1
        x_best = f.x0[0].flatten()
    
    else:

        # n_rs = int(min(100,max(f_eval_*.05,5)))       # iterations to find good starting point
        n_rs = int(max(x_dim+1,f_eval_*.05))
        iter_          = f_eval_ - n_rs
        # evaluate first point
        f_best, x_best = Random_search(f, x_dim, bounds, n_rs)

    opt = minimize(
        f.fun_test, 
        x_best, 
        bounds=bounds, 
        options = {'maxfev': iter_, 'disp': False},
        )
    
    team_names = ['9','10']
    cids = ['01234567']
    # return opt.x, opt.fun, team_names, cids, X_opt_plot, TR_l, xnew, backtrck_l, samples_number
    return opt.x, opt.fun, team_names, cids