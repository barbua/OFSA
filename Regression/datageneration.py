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
    n = gen_dat["n"]
    p = gen_dat["p"]
    k = gen_dat["k"]
    alpha = gen_dat["alpha"]
    beta_star = gen_dat["beta_star"]
    dat_type = gen_dat["dat_type"]
    if dat_type == 1:
        # set initial value
        X = np.zeros((n, p))
        beta = np.zeros((p, 1))
        itrue = np.array(range(k)) * 10
        beta[itrue] = beta_star            
        # generate data
        
        for i in range(n):
            Xtemp = np.random.randn(1, p)
            z = np.random.randn()
            Xtemp = alpha * z + Xtemp
            X[i] = Xtemp
        
        Y = X.dot(beta)
        Y = Y.T + np.random.randn(1, n)
        return(X, Y.T, beta, itrue)
    
    elif dat_type == 2:
        #set initial value
        delta = 0.9  ## 0.5 have a better result
        sigma_mat = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                sigma_mat[i, j] = delta**abs(i-j)
        # decomposition
        eig_vector, V = np.linalg.eig(sigma_mat)
        sigma_mat_sqrt = V.dot(np.diag(np.sqrt(eig_vector))).dot(V.T)
        ##
        ## Generate X
        ##
        X = np.random.randn(n, p)
        X = X.dot(sigma_mat_sqrt)
        ##
        beta = np.zeros((p, 1))
        itrue = np.arange(k) * 10
        beta[itrue] = beta_star
        ## Generate Y
        Y = X.dot(beta)
        Y = Y.T + np.random.randn(1, n)
        return(X, Y.T, beta, itrue)
########################################################################################









