# -*- coding: utf-8 -*-
##############################
### Lizhe Sun
### OLasso_cls, OMCP_cls
##############################
################
### Load Package
################
#import datageneration
import numpy as np
import onlineFSA
##############################
########################################################################################
########################################################################################
def MCP_threshold(beta, b, lbd):
    ####################
    p = beta.shape[0]
    ####################
    for j in range(p):
        if np.abs(beta[j, 0]) <= lbd:
            beta[j, 0] = 0
        elif (np.abs(beta[j, 0]) > lbd) and (np.abs(beta[j, 0]) <= b * lbd):
            beta[j, 0] = (beta[j, 0] - np.sign(beta[j, 0]) * lbd) / (1 - 1 / b)
        else:
            beta[j, 0] = beta[j, 0]
    ####################
    return(beta)
########################################################################################
########################################################################################
def onlineMCP_cls(XX, XY, n, lbd, eta, b, T):
    ###############################
    ### initial value
    ###############################
    p = XX.shape[0]
    beta_init = np.zeros((p, 1))
    ###############################
    ### Loss_list
    ###############################
    for t in range(T):
        grad_beta = onlineFSA.gradient_l2(XX, XY, beta_init) / n 
        beta = beta_init - eta * grad_beta
        beta_MCP = MCP_threshold(beta, b, eta * lbd)
        ###########################
        if np.linalg.norm(beta_MCP - beta_init) <= 1e-5:
            break
        ###########################
        beta_init = beta_MCP
    ###############################
    return(beta_MCP)
########################################################################################
########################################################################################
############################################
###online Lasso
############################################
def onlineLasso_cls(XX, XY, n, lbd, eta, T):
    ###
    ### setup initial value
    ###
    p = XX.shape[0]
    thr = eta * lbd
    beta = np.zeros((p, 1))
    beta_matrix = np.zeros((p, T))
    ###
    for t in range(T):
        ### initial gradient descent
        gradloss = onlineFSA.gradient_l2(XX, XY, beta)
        gradloss = eta * gradloss / (2 * n)
        for j in range(p):
            beta[j, 0] = beta[j, 0] - gradloss[j, 0]
            if beta[j, 0] > thr:
                beta[j, 0] = beta[j, 0] - thr
            elif beta[j, 0] < - thr:
                beta[j, 0] = beta[j, 0] + thr
            else:
                beta[j, 0] = 0
            ###
            ###
            beta_matrix[j, t] = beta[j, 0]
        #####
        if np.linalg.norm(beta_matrix[:,t] - beta_matrix[:,t-1]) <= 1e-6:
            break
    return beta