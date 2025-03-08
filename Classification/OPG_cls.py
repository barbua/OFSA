# -*- coding: utf-8 -*-
####################################
### Lizhe Sun  
### Online proximal gradient (For Classification)
####################################
##################
##Load Package
##################
#import eqcorrdata
import numpy as np
##################
##################
### one step OPG (Classification)
##################
################################
#### Logistic Gradient
################################
########################################################################################
########################################################################################
def grad_logistic(sample, label, beta, beta0, lbd_l2):
    '''
    Loss Function: Logistic Loss
    '''
    n, p = sample.shape
    temp1 = np.exp(- label * (beta0 * np.ones((n, 1)) + sample.dot(beta)))
    temp2 = label * temp1 / (np.ones((n, 1)) + temp1)
    grad_beta = - temp2.T.dot(sample) / n + lbd_l2 * beta.T
    grad_beta = grad_beta.T
    grad_beta0 = - np.sum(temp2) / n
    #######################################
    return(grad_beta, grad_beta0)
########################################################################################    
########################################################################################
###############
### OPG_lasso
###############
def OPG_Lasso(Xbatch, Ybatch, beta, beta0, lbd, lbd_l2, eta, mb_size):
    n, p = Xbatch.shape
    thr = lbd * eta
    N_iter = int(n / mb_size)
    ##########################################
    for i in range(N_iter):
        index = np.arange(i * mb_size, (i+1) * mb_size, 1)
        X_mb = Xbatch[index, :].reshape(mb_size, p)
        Y_mb = Ybatch[index, :].reshape(mb_size, 1)
        grad_beta, grad_beta0 = grad_logistic(X_mb, Y_mb, beta, beta0, lbd_l2) 
        ######################################
        #################
        ### update beta
        #################
        beta = beta - eta * grad_beta
        #################
        for j in range(p):
            if beta[j, 0] > thr:
                beta[j, 0] = beta[j, 0] - thr
            elif beta[j, 0] < - thr:
                beta[j, 0] = beta[j, 0] + thr
            else:
                beta[j, 0] = 0
        ########################
        ########################
        ### update beta0
        ########################
        beta0 = beta0 - eta * grad_beta0
        ########################
    return(beta, beta0)
########################################################################################
#gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":1}
#Xtr, Ytr, betastar_vec, istar = eqcorrdata.eqcorrdat_cls(gen_data)
##############
#beta = np.zeros((1000, 1))
#beta0 = 0
##############
#beta, beta0 = OPG_Lasso(Xtr, Ytr, beta, beta0, 0.13, 0.01, 25)
#print(np.flatnonzero(beta).shape[0])
########################################################################################
