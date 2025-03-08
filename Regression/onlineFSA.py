# -*- coding: utf-8 -*-
########################################
### Online Learning by Feature Selection with Annealing Algorithm
### including function: 
### 1. computing the running sums 
### 2. adding running sums
### 3. standardize running sums
### 4. onlineFSA
### 5. OLS regression
### author: Lizhe
#######################################
#### Load package
#######################################
import math
import numpy as np
##########################################################################
##
## Define the running sum function
##
###########################################################################
###########################################################################
#### Introduction
#### input data set X, Y. X is n by p data matrix and Y is n by 1 response
#### Output is running sums: Sx, Sy, Sxx, Sxy,
###########################################################################
def running_aves(X, Y):
    ##initial value
    n, p = X.shape
    Sx = np.array(np.sum(X, axis = 0))
    Sy = np.sum(Y.T)
    ## Array Sx
    Sx = np.array([Sx])
    Sxx = X.T.dot(X)
    Sxy = X.T.dot(Y)
    Syy = Y.T.dot(Y)
    ##
    runningaves = {"n":n, "Sx":Sx, "Sy":Sy, "Sxx":Sxx, "Sxy":Sxy, "Syy":Syy}
    ##
    return(runningaves)


################################################################
##  Define Add running averages
################################################################
##                        Introduction
################################################################
def add_runningaves(ra1, ra2):
    ##
    n = ra1["n"] + ra2["n"]
    Sx = ra1["Sx"] + ra2["Sx"]
    Sy = ra1["Sy"] + ra2["Sy"]
    Sxx = ra1["Sxx"] + ra2["Sxx"]
    Sxy = ra1["Sxy"] + ra2["Sxy"]
    Syy = ra1["Syy"] + ra2["Syy"]
    added_runningaves = {"n":n, "Sx":Sx, "Sy":Sy, "Sxx":Sxx, "Sxy":Sxy,"Syy":Syy}
    ##
    return(added_runningaves)


################################################################
###### Standardize Runningsums
###### parameters: n, Sx, Sy, Sxx, Sxy
################################################################
def standardize_ra(runningaves):
    ###
    n = runningaves["n"]
    Sx = runningaves["Sx"]
    Sy = runningaves["Sy"]
    Sxx = runningaves["Sxx"]
    Sxy = runningaves["Sxy"]
    ##
    ##
    p = Sxx.shape[0]
    ##
    ##
    mu_x = Sx / n
    mu_y = Sy / n
    ## S_x^2
    Sx_sq = np.diag(Sxx)
    ## sigma_sqx vector
    sigmasq_x = Sx_sq / n - (Sx / n)**2
    ## sigma_x
    std_x = np.sqrt(sigmasq_x)
    #####
    inv_sigma = 1 / std_x
    ##
    ## Calculate the running sum and standardize running sum
    ##
    ## XY_normalize
    ##
    Temp1 = inv_sigma.T * (Sxy)
    Temp2 = mu_y * inv_sigma * Sx             
    XY_normalize = Temp1 - Temp2.T
    ##
    ## XX_normalize
    ##
    XX_normalize = Sxx - Sx.T.dot(Sx) / n
    ##
    ##
    for i in range(p):
        XX_normalize[i, :] = XX_normalize[i, :] * inv_sigma[0, i]
    for i in range(p):
        XX_normalize[:, i] = XX_normalize[:, i] * inv_sigma[0, i]
    ##
    ##
    ##                   
    return(XX_normalize, XY_normalize, mu_x, mu_y, std_x)

######################################################################
##          Define gradient for Loss function
######################################################################
def gradient_l2(XX, XY, beta):
    grad = - XY + XX.dot(beta)
    return grad
############################################################################
####                     FSA Algorithm              ########################
#### FSA_para : n, k, eta, mu, N_iter, shrinkage:lbd
############################################################################
def onlineFSA(XX, XY, FSA_para, pre_train):
    ###
    n = FSA_para["n"]
    k = FSA_para["k"]
    eta = FSA_para["eta"]
    mu = FSA_para["mu"]
    lbd = FSA_para["lbd"]
    N_iter = FSA_para["N_iter"]
    ## setup initial value
    p = XX.shape[0]
    mom = 0.9
    ## To calculate and update 
    beta = np.zeros((p,1))
    dbeta = np.zeros((p,1))
    ##
    sel = np.arange(p)
    XX_sel = XX
    XY_sel = XY
    beta_sel = beta
    dbeta_sel = dbeta
    ##
    loop_list = np.arange(-pre_train, N_iter+1)
    ##
    ## Loop
    ##
    for i in loop_list:
        ###
        ### Calculate gradloss
        ###
        grad = gradient_l2(XX_sel, XY_sel, beta_sel)
        dbeta_sel = mom * dbeta_sel - (1 - mom) * eta * (grad / n + lbd * beta_sel) 
        beta_sel = beta_sel + dbeta_sel
        ###
        ###
        if beta_sel.shape[0] > k:
            ###
            prc = min(1, (N_iter - i)/(i * mu + N_iter))
            M_e = math.floor(k + (p - k) * max(0, prc))
            ### Rank the beta, increase
            ### Get index of abs(beta)
            beta_index = np.argsort(abs(beta_sel.T), axis = 1)
            beta_index = beta_index.ravel()
            ###
            ### Reverse the index, decreasing
            beta_dec_index = beta_index[::-1]           
            sel_index = beta_dec_index[0:M_e]
            sel_index = np.sort(sel_index)
            sel = sel[sel_index]
            ###
            ### Update beta_sel and dbeta_sel
            ###
            beta_sel = beta_sel[sel_index]
            dbeta_sel = dbeta_sel[sel_index]
            XX_sel = XX[np.ix_(sel, sel)]
            XY_sel = XY[sel]
    #####   
    #####
    return (beta_sel, sel)


####################################################################
#### runningsums_OLS
#### lambda is shrinkage
#####################################################################
def OLS_runningaves(XX, XY, lbd):
    p = XX.shape[0]
    XX_temp = XX + lbd * np.identity(p)
    XX_inv = np.linalg.inv(XX_temp)
    beta_ols = XX_inv.dot(XY)
    return (beta_ols)
    