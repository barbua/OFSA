# -*- coding: utf-8 -*-
######################################
### Data generation (Runningaves)
######################################
################
## Load Package
################
import numpy as np
#############################################################################
######################     data generation                  #################
#############################################################################
##### Equal correlated data
#############################################################################
def eqcorrdat_cls(gen_dat):
    n = gen_dat["n"]
    p = gen_dat["p"]
    k = gen_dat["k"]
    alpha = gen_dat["alpha"]
    beta_star = gen_dat["beta_star"]
    #################################################
    ### set initial value
    #################################################
    X = np.zeros((n, p))
    beta = np.zeros((p, 1))
    itrue = np.array(range(k)) * 10
    beta[itrue] = beta_star            
    ### generate data
    for i in range(n):
        X_temp = np.random.randn(1, p)
        z = np.random.randn()
        X_temp = alpha * z + X_temp
        X[i] = X_temp
    ################################################
    Y = X.dot(beta)
    Y = Y.T + np.random.randn(1, n)
    Y = np.sign(Y)
    return(X, Y.T, beta, itrue)
########################################################################################
########################################################################################
########################################################################################
def eqcorrdat_reg(gen_dat):
    n = gen_dat["n"]
    p = gen_dat["p"]
    k = gen_dat["k"]
    alpha = gen_dat["alpha"]
    beta_star = gen_dat["beta_star"]
    #################################################
    ### set initial value
    #################################################
    X = np.zeros((n, p))
    beta = np.zeros((p, 1))
    itrue = np.array(range(k)) * 10
    beta[itrue] = beta_star            
    ### generate data
    for i in range(n):
        X_temp = np.random.randn(1, p)
        z = np.random.randn()
        X_temp = alpha * z + X_temp
        X[i] = X_temp
    ################################################
    Y = X.dot(beta)
    Y = Y.T + np.random.randn(1, n)
    return(X, Y.T, beta, itrue)
########################################################################################











