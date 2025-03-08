# -*- coding: utf-8 -*-
##############################
### Lizhe Sun
### Gradient of Loss Function
##############################
################
### Load Package
################
import numpy as np
###############################
#### Gradient of Loss Function
#### Loss with L2 penalty
###############################   
def gradlogistic(X, Y, w, w0, lbd):
    #######################################
    '''
    Loss Function: Logistic Loss with L2 penalty
    '''
    n, p = X.shape
    temp1 = np.exp(- Y * (w0 * np.ones((n, 1)) + X.dot(w)))
    temp2 = Y * temp1 / (np.ones((n, 1)) + temp1)
    grad_w = - temp2.T.dot(X) / n + lbd * w.T
    grad_w = grad_w.T
    grad_w0 = np.sum(temp2) / n
    #######################################
    return(grad_w, grad_w0)
###
########################################################################################
###
########################################################################################
###
def gradlogistic_nointercept(X, Y, w, lbd):
    #######################################
    '''
    Loss Function: Logistic Loss with L2 penalty
    '''
    n, p = X.shape
    temp1 = np.exp(- Y * X.dot(w))
    temp2 = Y * temp1 / (np.ones((n, 1)) + temp1)
    grad_w = - temp2.T.dot(X) / n + lbd * w.T
    grad_w = grad_w.T
    #######################################
    return(grad_w)
    
###
########################################################################################
###
########################################################################################
###    
def gradreg(X, Y, w, w0, lbd):
    ######################################
    '''
    Loss function: l2 loss with L2 penalty
    '''
    n, p = X.shape
    temp1 = Y - w * np.ones((n, 1)) - X.dot(w)
    grad_w = - X.T.dot(temp1) / n + lbd * w
    grad_w0 = np.sum(temp1) / n
    ######################################
    return(grad_w, grad_w0)
###
########################################################################################
###
########################################################################################
###
def gradlorenz(X, Y, w, w0, lbd):
    ##########################################
    '''
    Loss function: Lorenz loss with L2 penalty
    '''
    n, p = X.shape
    Xw = Y * (X.dot(w) + w0 * np.ones((n, 1)))
    temp1 = Xw - np.ones((n, 1))
    index1 = np.where(temp1.ravel > 0)[0]
    temp1[index1,:] = 0
    temp2 = np.ones((n, 1)) + temp1**2
    temp3 = 2 * Y * temp1 / temp2
    grad_w = temp3.T.dot(X)
    grad_w = grad_w.T
    grad_w = grad_w / n + lbd * w
    grad_w0 = np.sum(temp3) / n 
    ###########################################
    return(grad_w, grad_w0)
###
########################################################################################
###
########################################################################################
###
########################################################################################
###
########################################################################################
###
def hard_threshold(w, thr):
    ###
    increa_index = np.argsort(abs(w.T), axis = 1)
    increa_index = increa_index.ravel()
    ### Reverse the index, decreasing
    decrea_index = increa_index[::-1]           
    sel_index = decrea_index[0:thr]
    sel_index = np.sort(sel_index)
    w = w[sel_index]
    ###
    return (w, sel_index)
########################################################################################

    
        
        
        
    
