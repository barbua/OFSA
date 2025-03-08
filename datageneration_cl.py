# -*- coding: utf-8 -*-
###############################
## OnlineFSA
## Lizhe Sun
###############################
################
## Load Package
################
import numpy as np
########################################################################################
###################### data generation for incremental data     ########################
########################################################################################
def generate_data(gen_dat):
    datat=np.float32
    n = gen_dat["n"]
    p = gen_dat["p"]
    k = gen_dat["k"]
    alpha = gen_dat["alpha"]
    beta_star = gen_dat["beta_star"]
    #################################################
    ### set initial value
    #################################################
    X = np.zeros((n, p),dtype=datat)
    beta = np.zeros((p, 1),dtype=datat)
    itrue = np.array(range(k)) * int(p/k)
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
