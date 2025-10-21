# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 15:12:20 2023

@author: adelrioc
"""

from scipy.optimize import minimize
from scipy.optimize import direct, Bounds
import scipy
import numpy as np

#########################
# --- Random search --- #
#########################

# (f, N_x: int, bounds: array[array[float]], N: int = 100) -> array(N_X), float 
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

def opt_Powell(f, x_dim, bounds, iter_tot):
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

####################
# COBYLA algorithm #
####################

def opt_COBYLA(f, x_dim, bounds, iter_tot, has_x0 =False):
    '''
    params: parameters that define the rbf model
    X:      matrix of previous datapoints
    iter_tot: evaluation budget, given by f_eval_
    '''
    if has_x0 == True:

        iter_          = iter_tot - 1
        x_best = f.x0[0].flatten()

    else:
        # n_rs = int(min(100,max(iter_tot*.05,5))) # old

        # iterations to find good starting point
        n_rs = int(max(x_dim+1,iter_tot*.05))
        
        # evaluate first point
        f_best, x_best = Random_search(f, x_dim, bounds, n_rs)
        iter_          = iter_tot - n_rs

    opt = minimize(
        f.fun_test, 
        x_best, 
        method='COBYLA',
        options={'maxiter': iter_}) 

    team_names = ['9','10']
    cids = ['01234567']
    return opt.x, opt.fun, team_names, cids

########################
# NelderMead algorithm #
########################

def opt_NelderMead(f, x_dim, bounds, iter_tot):
    '''
    params: parameters that define the rbf model
    X:      matrix of previous datapoints
    '''

    n_rs = int(min(100,max(iter_tot*.05,5)))       # iterations to find good starting point

    # evaluate first point
    f_best, x_best = Random_search(f, x_dim, bounds, n_rs)
    iter_          = iter_tot - n_rs

    opt = minimize(f.fun_test, x_best, method='Nelder-Mead', 
                   options={'maxfev': iter_}) 
    
    team_names = ['11','12']
    cids = ['01234567']
    return opt.x, opt.fun, team_names, cids

##########################
# Differential Evolution #
##########################

def opt_DE(f, x_dim, bounds, iter_tot):
    '''
    params: parameters that define the rbf model
    X:      matrix of previous datapoints
    '''

    # evaluate first point
    popsize_ = int(min(100,max(iter_tot*.05,5))) 
    maxiter_ = int(iter_tot/popsize_) + 1

    opt = scipy.optimize.differential_evolution(f.fun_test, bounds, 
                                                maxiter=maxiter_, 
                                                popsize=popsize_) 

    team_names = ['13','14']
    cids = ['01234567']
    return opt.x, opt.fun, team_names, cids

################
# Basinhopping #
################

def opt_Basinhopping(f, x_dim, bounds, iter_tot):
    '''
    params: parameters that define the rbf model
    X:      matrix of previous datapoints
    '''

    n_rs = int(min(100,max(iter_tot*.05,5)))       # iterations to find good starting point

    # evaluate first point
    f_best, x_best   = Random_search(f, x_dim, bounds, n_rs)
    iter_            = iter_tot - n_rs
    minimizer_kwargs = {"method": "BFGS"}
    niter_           = int(iter_tot/3)

    opt = scipy.optimize.basinhopping(f.fun_test, x_best, 
                                      minimizer_kwargs=minimizer_kwargs) 

    team_names = ['15','16']
    cids = ['01234567']
    return opt.x, opt.fun, team_names, cids

##########
# Direct #
##########

def opt_Direct(f, x_dim, bounds, iter_tot):
    '''
    params: parameters that define the rbf model
    X:      matrix of previous datapoints
    '''
    
    bounds = bounds.tolist()
    opt    = scipy.optimize.direct(f.fun_test, bounds, maxfun=iter_tot) 

    team_names = ['17','18']
    cids = ['01234567']
    return opt.x, opt.fun, team_names, cids




















