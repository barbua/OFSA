# -*- coding: utf-8 -*-
#########################################
## Online Lasso and Online Adaptive Lasso
## Lizhe Sun
#########################################
###################################
### Load Package
###################################
import numpy as np
import onlineFSA
############################################
###online Lasso
###########################################
def online_lasso(XX, XY, N, lbd, eta, T):
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
        gradloss = eta * gradloss / (2 * N)
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

#############################################################################
### Online Adaptive Lasso
#############################################################################
def online_adalasso(XX, XY, n, lbd, eta, gamma, T):
    ####################
    ### initial value
    ####################
    p = XX.shape[0]
    thr = eta * lbd
    beta_adalasso = np.zeros((p, 1))
    beta_mat = np.zeros((p, T))
    if n <= p:
        beta_ols = onlineFSA.OLS_runningsums(XX, XY, 0.01)
        beta_ols = beta_ols.ravel()
    else:
        beta_ols = onlineFSA.OLS_runningsums(XX, XY, 0)
        beta_ols = beta_ols.ravel()
    ###########   
    weight = 1 / np.abs(beta_ols)**gamma
    inv_weight = 1 / weight
    inv_weight_mat = np.diag(inv_weight)
    #########
    ## update running sums
    #########
    XX_weight = inv_weight_mat.dot(XX).dot(inv_weight_mat)
    XY_weight = inv_weight_mat.dot(XY)
    ####
    for t in range(T):
        #####
        ### update gradient descent
        #####
        gradloss = onlineFSA.gradient_l2(XX_weight, XY_weight, beta_adalasso)
        gradloss = eta * gradloss / (2 * n)
        for j in range(p):
            beta_adalasso[j, 0] = beta_adalasso[j, 0] - gradloss[j, 0]
            if beta_adalasso[j, 0] > thr:
                beta_adalasso[j, 0] = beta_adalasso[j, 0] - thr
            #####
            elif beta_adalasso[j, 0] < - thr:
                beta_adalasso[j, 0] = beta_adalasso[j, 0] + thr
                
            else:
                beta_adalasso[j, 0] = 0
            ####
            ####
            beta_mat[j, t] = beta_adalasso[j, 0]
        ####
        if np.linalg.norm(beta_mat[:,t] - beta_mat[:,t-1]) <= 1e-6:
            break
    ####
    return beta_adalasso

#########################################################################################
#def online_AdaLassoADMM(XX, XY, N, lbd, rho, gamma, T):
#    ##############
#    ###initial value
#    ##############
#    p = XX.shape[0]
#    thr = lbd / rho
#    ###################################################################
#    if N <= p:
#        beta_ols = onlineFSA.OLS_runningsums(XX, XY, 0.01)
#        beta_ols = beta_ols.ravel()
#    else:
#        beta_ols = onlineFSA.OLS_runningsums(XX, XY, 0)
#        beta_ols = beta_ols.ravel()
#    ###########   
#    weight = 1 / np.abs(beta_ols)**gamma
#    inv_weight = 1 / weight
#    inv_weight_mat = np.diag(inv_weight)
#    #########
#    ## update running sums
#    #########
#    XX_weight = inv_weight_mat.dot(XX).dot(inv_weight_mat)
#    XY_weight = inv_weight_mat.dot(XY)
#    ####
#    XX_temp = np.linalg.inv(XX_weight + rho * np.eye(p))
#    theta0 = np.zeros((p, 1))
#    mu0 = np.zeros((p, 1))
#    ###############################################################
#    for i in range(T):
#        #############
#        ### ADMM update rule
#        #############
#        beta = XX_temp.dot(XY_weight + rho * theta0 - mu0)
#        theta = beta + mu0 / rho
#        for j in range(p):
#            if theta[j, 0] > thr:
#                theta[j, 0] = theta[j, 0] - thr
#            elif theta[j, 0] < - thr:
#                theta[j, 0] = theta[j, 0] + thr
#            else:
#                theta[j, 0] = 0
#        mu = mu0 + rho * (beta - theta)
#        theta0 = theta
#        mu0 = mu
#        #############
#        ### stop rule
#        #############
#        if np.linalg.norm(beta - theta) <= 1e-6:
#            break
#        ####
#    return theta



########################################################################################
def online_elnet(XX, XY, N, lbd1, lbd2, eta, T):
    #### set the initial value
    p = XX.shape[0]
    thr = eta * lbd1
    beta_elnet = np.zeros((p, 1))
    beta_mat = np.zeros((p, T))
    ####
    for t in range(T):
        #######
        ## update gradient
        #######
        gradloss = onlineFSA.gradient_l2(XX, XY, beta_elnet)
        gradloss = eta * gradloss / (2 * N)
        for j in range(p):
            ### update each element beta
            beta_elnet[j, 0] = beta_elnet[j, 0] - gradloss[j, 0] - eta * lbd2 * beta_elnet[j, 0]
            if beta_elnet[j, 0]  > thr:
                beta_elnet[j, 0] = beta_elnet[j, 0]  - thr
            #####
            elif beta_elnet[j, 0]  < - thr:
                beta_elnet[j, 0] = beta_elnet[j, 0]  + thr
            #####
            else:
                beta_elnet[j, 0] = 0
            ####
            beta_mat[j, t] = beta_elnet[j, 0]
        #####
        if np.linalg.norm(beta_mat[:, t] - beta_mat[:, t-1]) <= 1e-6:
            break
    ####
    return beta_elnet

# =============================================================================
# 
# =============================================================================

def elnet_SCD(XX, XY, N, lbd1, lbd2, T):
    ### set the initial value
    p = XX.shape[0]
    beta = np.zeros((p, 1))
    beta_mat = np.zeros((p, T))
    ### initial gradient descent
    gradloss = onlineFSA.gradient_l2(XX, XY, beta)
    gradloss = gradloss / (2 * N)
    ### SCD Algorithm
    for t in range(T):
        ### initial the feature list
        feature_list = np.random.permutation(p)
        for j in range(p):
            rsel = feature_list[j]
            beta[rsel, 0] = beta[rsel, 0] - gradloss[rsel, 0]
            if beta[rsel, 0] > lbd1:
                beta[rsel, 0] = beta[rsel, 0] - lbd1
                beta[rsel, 0] = beta[rsel, 0] / (1 + lbd2)
            elif beta[rsel, 0] < - lbd1:
                beta[rsel, 0] = beta[rsel, 0] + lbd1
                beta[rsel, 0] = beta[rsel, 0] / (1 + lbd2)
            else:
                beta[rsel, 0] = 0
            ####
            beta_mat[rsel, t] = beta[rsel, 0]
            #####
            ### update gradient 
            #####
            XX_vec = XX[:,rsel].reshape(p, 1)
            gradloss = gradloss + XX_vec * (beta_mat[rsel, t] - beta_mat[rsel, t-1]) / N
        ########
        if np.linalg.norm(beta_mat[:, t] - beta_mat[:, t-1]) <= 1e-6:
            break
    ######    
    return beta
    
#########################################################################################
#############
#### Stochastic Coordinate Decsent (SCD Algorithm)
#############
#def SCD_Lasso(XX, XY, N, lbd, T):
#    ### set the initial value
#    p = XX.shape[0]
#    beta = np.zeros((p, 1))
#    beta_mat = np.zeros((p, T))
#    ### initial gradient descent
#    gradloss = onlineFSA.gradloss_func(XX, XY, beta, 0)
#    gradloss = gradloss / (2 * N)
#    ### SCD Algorithm
#    for t in range(T):
#        ### initial the feature list
#        feature_list = np.random.permutation(p)
#        for j in range(p):
#            rsel = feature_list[j]
#            beta[rsel, 0] = beta[rsel, 0] - gradloss[rsel, 0]
#            if beta[rsel, 0] > lbd:
#                beta[rsel, 0] = beta[rsel, 0] - lbd
#            elif beta[rsel, 0] < - lbd:
#                beta[rsel, 0] = beta[rsel, 0] + lbd
#            else:
#                beta[rsel, 0] = 0
#            ####
#            beta_mat[rsel, t] = beta[rsel, 0]
#            #####
#            ### update gradient 
#            #####
#            XX_vec = XX[:,rsel].reshape(p, 1)
#            gradloss = gradloss + XX_vec * (beta_mat[rsel, t] - beta_mat[rsel, t-1]) / N
#        ########
#        if np.linalg.norm(beta_mat[:, t] - beta_mat[:, t-1]) <= 1e-6:
#            break
#    ######    
#    return beta
#
            
#############
#### Stochastic Coordinate Decsent (SCD Algorithm)
#############
#def SCD1_Lasso(XX, XY, N, lbd, T):
#    ### set the initial value
#    p = XX.shape[0]
#    beta = np.zeros((p, 1))
#    beta_mat = np.zeros((p, T))
#    ### initial gradient descent
#    gradloss = onlineFSA.gradloss_func(XX, XY, beta, 0)
#    gradloss = gradloss / (2 * N)
#    ### SCD Algorithm
#    for t in range(T):
#        ### initial the feature list
#        feature_list = np.random.permutation(p)
#        for j in range(p):
#            rsel = feature_list[j]
#            beta[rsel, 0] = beta[rsel, 0] - gradloss[rsel, 0]
#            if beta[rsel, 0] > lbd:
#                beta[rsel, 0] = beta[rsel, 0] - lbd
#            elif beta[rsel, 0] < - lbd:
#                beta[rsel, 0] = beta[rsel, 0] + lbd
#            else:
#                beta[rsel, 0] = 0
#            #####
#            ### update gradient 
#            #####
#            gradloss = onlineFSA.gradloss_func(XX, XY, beta, 0)
#            gradloss = gradloss / (2 * N)
#            ####
#            beta_mat[rsel, t] = beta[rsel, 0]
#        ########
#        if np.linalg.norm(beta_mat[:, t] - beta_mat[:, t-1]) <= 1e-6:
#            break
#    ######    
#    return beta
#
#   


















