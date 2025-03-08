# -*- coding: utf-8 -*-
##############################
### Yangzi Guo
### First and second order truncated online feature selection utilities.
##############################
################
### Load Package
################
import numpy as np
#import eqcorrdata
###############################
#### hinge and hinge squared Gradient
########################################################################################    
########################################################################################
########################################################################################
def grad_hinge(sample, label, beta, beta0, lbd):
    '''
    Hinge Loss with L2 penalty
    '''
    n, p = sample.shape
    Xbeta = sample.dot(beta) + beta0 * np.ones((n, 1))
    temp1 = np.ones((n, 1)) - label * Xbeta
    index1 = temp1 > 0
    grad_beta = - np.sum(label * sample * index1, axis = 0) / n 
    grad_beta = grad_beta.reshape(p, 1) + lbd * beta
    grad_beta0 = - np.sum(label*index1) / n
    return(grad_beta, grad_beta0)
########################################################################################    
########################################################################################
########################################################################################    
def grad_sq_hinge(sample, label, beta, sigma, lbd):
    '''
    Squared Hinge Loss with penalty
    '''
    n, p = sample.shape
    Xbeta = sample.dot(beta)         # nxp dot px1 = nx1
    temp = np.ones((n, 1)) - label * Xbeta   # nx1 * nx1 = nx1
    temp = temp * (temp > 0)   # nx1 * nx1 = nx1
    gradbeta_sq_hinge = - 2 * temp * label * sample   # nx1 * nxp * nx1 = nxp
    mu = 1.0 / (np.dot(sample * sample, sigma) + lbd)
    beta = beta - 0.5 * np.mean(mu * gradbeta_sq_hinge * sigma.T, axis = 0).reshape(-1,1)
    diag_xx = np.mean(sample * sample, axis=0).reshape(-1,1) / lbd
    sigma = 1.0/(1.0 / sigma + diag_xx)
    return(beta, sigma)
########################################################################################
########################################################################################
########################################################################################
def truncate1(beta, k):
    ###
    increa_index = np.argsort(abs(beta.T), axis = 1)
    increa_index = increa_index.ravel()
    ### Reverse the index, decreasing
    decrea_index = increa_index[::-1]
    sel_index = decrea_index[0:k]
    zero_index = decrea_index[k:]
    beta[zero_index] = 0.0
    ###
    return (beta, sel_index)    
########################################################################################
########################################################################################
def truncate2(beta, sigma, k):
    ###
    increa_index = np.argsort(abs(sigma.T), axis = 1)
    increa_index = increa_index.ravel()
    ### Reverse the index, decreasing
    sel_index = increa_index[0:k]
    beta_sel = beta[sel_index]
    ###
    return (beta_sel, sel_index)    
########################################################################################
########################################################################################
def OFTSGD_cls(Xbatch, Ybatch, beta, beta0, k, lbd, eta, mb_size):
    ##########################################
    n, p = Xbatch.shape
    N_iter = int(n / mb_size)
    ##########################################
    for i in range(N_iter):
        index = np.arange(i * mb_size, (i+1) * mb_size, 1)
        X_mb = Xbatch[index, :].reshape(mb_size, p)
        Y_mb = Ybatch[index, :].reshape(mb_size, 1)
        grad_beta, grad_beta0 = grad_hinge(X_mb, Y_mb, beta, beta0, lbd)
        beta = beta - eta * grad_beta
        beta0 = beta0 - eta * grad_beta0
        beta = min(1.0,(1.0 / np.sqrt(lbd)) / np.linalg.norm(beta)) * beta   # Sparse projection
        ###
        ### Truncate k
        ###
        beta, sel = truncate1(beta, k)
        ###
    return(beta, beta0, sel)
########################################################################################
########################################################################################
def OSTSGD_cls(Xbatch, Ybatch, beta, k, lbd, sigma, mb_size):
    ##########################################
    n, p = Xbatch.shape
    N_iter = int(n / mb_size)
    ##########################################
    for i in range(N_iter):
        index = np.arange(i * mb_size, (i+1) * mb_size, 1)
        X_mb = Xbatch[index, :].reshape(mb_size, p)
        Y_mb = Ybatch[index, :].reshape(mb_size, 1)
        beta, sigma = grad_sq_hinge(X_mb, Y_mb, beta, sigma, lbd)
        ###
        ### Truncate k
        ###
        beta0 = beta[-1]
        beta_sel, sel = truncate2(beta, sigma[0:p], k)
        ###
        beta = np.zeros((p, 1))
        beta[sel] = beta_sel
        beta[-1] = beta0
    ##########################################
    return(beta, sel)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
