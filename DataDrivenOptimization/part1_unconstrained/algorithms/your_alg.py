from scipy.optimize import minimize
import numpy as np

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


####################
# Powell algorithm #
####################

def your_alg(f, x_dim, bounds, iter_tot):
    '''
    params: parameters that define the rbf model
    X:      matrix of previous datapoints
    '''

    n_rs = int(min(100,max(iter_tot*.05,5)))       # iterations to find good starting point

    # evaluate first point
    f_best, x_best = Random_search(f, x_dim, bounds, n_rs)
    iter_          = iter_tot - n_rs

    opt = minimize(f.fun_test, x_best, bounds=bounds, method='Powell', 
                    options={'maxfev': iter_}) 

    team_names = ['7','8']
    cids = ['01234567']
    return opt.x, opt.fun, team_names, cids