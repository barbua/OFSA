# -*- coding: utf-8 -*-
####################################
### Lizhe Sun  
### Dual Averaging Algorithms (For Classification)
####################################
##################
##Load Package
##################
#import eqcorrdata
import numpy as np
##################
##################
### one step RDA (Classification)
##################
###############################
#### Logistic Gradient
################################
########################################################################################
########################################################################################
def grad_logistic(sample, label, beta, beta0):
    '''
    Loss Function: Logistic Loss with L2 penalty
    '''
    n, p = sample.shape
    temp1 = np.exp(- label * (beta0 * np.ones((n, 1)) + sample.dot(beta)))
    temp2 = label * temp1 / (np.ones((n, 1)) + temp1)
    grad_beta = - temp2.T.dot(sample) / n
    grad_beta = grad_beta.T
    grad_beta0 = - np.sum(temp2) / n
    #######################################
    return(grad_beta, grad_beta0)
########################################################################################    
########################################################################################
########################################################################################
###############
### RDA_lasso
###############
def RDA_elnet(Xbatch, Ybatch, beta, beta0, RDA_para, ada_num):
    n, p = Xbatch.shape
    lbd = RDA_para["lbd"]
    gamma = RDA_para["gamma"]
    mini_batch = RDA_para["mini_batch"]
    ###################################
    iter_time = int(n / mini_batch)
    ave_gradbeta = np.zeros((p, 1))
    ave_gradbeta0 = 0
    ###################################
    for i in range(iter_time):
        index = np.arange(i * mini_batch, (i + 1) * mini_batch)
        X_mb = Xbatch[index,:].reshape(mini_batch, p)
        Y_mb = Ybatch[index,:].reshape(mini_batch, 1)
        grad_beta, grad_beta0 = grad_logistic(X_mb, Y_mb, beta, beta0)
        ave_gradbeta = ((i + ada_num) / (i + 1 + ada_num)) * ave_gradbeta + (1 / (i + 1 + ada_num)) * grad_beta
        ave_gradbeta0 = ((i + ada_num) / (i + 1 + ada_num)) * ave_gradbeta0 + (1 / (i + 1 + ada_num)) * grad_beta0
        #######################
        ### update beta
        #######################
        for j in range(p):
            if ave_gradbeta[j, 0] > lbd:
                beta[j, 0] = - (ave_gradbeta[j, 0] - lbd) / gamma
            elif ave_gradbeta[j, 0] < -lbd:
                beta[j, 0] = - (ave_gradbeta[j, 0] + lbd) / gamma
            else:
                beta[j, 0] = 0
        ########################
        ### update beta0
        ########################
        beta0 = - ave_gradbeta0
        ########################
    return(beta, beta0)
########################################################################################
#gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":1}
#RDA_parameter = {"lbd":0.39, "gamma":100, "mini_batch":25}
#Xtr, Ytr, betastar_vec, istar = eqcorrdata.eqcorrdat_cls(gen_data)
##############
#beta = np.zeros((1000, 1))
#beta0 = 0
##############
#beta, beta0 = RDA_elnet(Xtr, Ytr, beta, beta0, RDA_parameter, 0)
#print(np.flatnonzero(beta).shape[0])         
