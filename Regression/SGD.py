# -*- coding: utf-8 -*-
#################################
## SGD_onlineFSA
## Lizhe Sun
#################################
##################
##Load Package
##################
import numpy as np
###################################################################
def grad_L2loss(datX, datY, beta, beta0):
    n, p = datX.shape
    temp1 = datY - beta0 * np.ones((n, 1)) - datX.dot(beta)
    grad_beta = - 2 * datX.T.dot(temp1) / n
    grad_beta0 = - 2 * np.sum(temp1) / n
    return(grad_beta, grad_beta0)
###################################################################
#####################################
## define feature selection function 
#####################################
def Hard_threshold(beta, sel_num):
    beta_index = np.argsort(abs(beta.T), axis = 1)
    beta_index = beta_index.ravel()
    ### Reverse the index, decreasing
    beta_dec_index = beta_index[::-1]           
    sel_index = beta_dec_index [0:sel_num]
    sel_index = np.sort(sel_index)
    beta_sel = np.zeros((beta.shape[0], 1))
    beta_sel[sel_index] = beta[sel_index]
    return (beta_sel, sel_index)
###################################################################
################## train_SGD and SGDth ############################
### Generate data and train SGD
###################################################################
def train_SGD(datX, datY, beta, beta0, eta, mb_size):
    n, p = datX.shape
    inner_loop = int(n / mb_size)
    for i in range(inner_loop):
        index = np.arange(mb_size * i, mb_size * (i+1), 1)
        X_mb = datX[index,:]
        Y_mb = datY[index,:]
        grad_beta, grad_beta0 = grad_L2loss(X_mb, Y_mb, beta, beta0)
        beta = beta - eta * grad_beta
        beta0 = beta0 - eta * grad_beta0
        ######################################
    return beta, beta0
########################################################################################
########################################################################################
def train_SIHT(datX, datY, beta, beta0, eta, fea_num, mb_size):
    n, p = datX.shape
    inner_loop = int(n / mb_size)
    for i in range(inner_loop):
        index = np.arange(mb_size * i, mb_size * (i+1), 1)
        X_mb = datX[index,:]
        Y_mb = datY[index,:]
        grad_beta, grad_beta0 = grad_L2loss(X_mb, Y_mb, beta, beta0)
        beta = beta - eta * grad_beta
        beta0 = beta0 - eta * grad_beta0
        beta, sel_index = Hard_threshold(beta, fea_num) 
    ####################################################################################    
    return beta, beta0, sel_index
########################################################################################
########################################################################################
####################
### Truncated Gradient
####################
def train_TGrad(datX, datY, beta, beta0, eta, lbd, mb_size):
    n, p = datX.shape
    inner_loop = int(n / mb_size)
    ################
    for i in range(inner_loop):
        index = np.arange(mb_size * i, mb_size * (i+1), 1)
        X_mb = datX[index,:]
        Y_mb = datY[index,:]
        grad_beta, grad_beta0 = grad_L2loss(X_mb, Y_mb, beta, beta0)
        beta = beta - eta * grad_beta
        beta0 = beta0 - eta * grad_beta0
        ################
        for j in range(p):
            if (beta[j,0] >= 0) and (beta[j,0] <= lbd):
                beta[j,0] = max(0, beta[j,0] - eta*lbd)
            elif (beta[j,0] >= -lbd) and (beta[j,0] < 0):
                beta[j,0] = min(0, beta[j,0] + eta*lbd)
            else:
                beta[j,0] = beta[j,0]
    ###############
    return beta, beta0
########################################################################################            
    
    