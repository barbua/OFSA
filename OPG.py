# -*- coding: utf-8 -*-
####################################
### Lizhe Sun  
### Online Promixal Gradient
####################################
##################
##Load Package
##################
import datageneration
import numpy as np
##################
##################
### one step OPG (Regression)
##################
def gradient_loss(X, Y, beta, mini_batch):
    temp1 = Y - X.dot(beta)
    gradloss = - 2 * X.T.dot(temp1)
    gradloss = gradloss / mini_batch
    return gradloss
####################################################################
####################################################################
#######
### OPG - Lasso
#######
def OPG_Lasso(datX, datY, beta, beta0, OPG_para):
    n, p = datX.shape
    lbd = OPG_para["lbd"]
    eta = OPG_para["eta"]
    eta1 = OPG_para["eta1"]
    mini_batch = OPG_para["mini_batch"]
    thr = lbd * eta1
    N_iter = int(n / mini_batch)
    ##########################################
    for i in range(N_iter):
        index = np.arange(i * mini_batch, (i+1) * mini_batch, 1)
        datX_minibatch = datX[index, :].reshape(mini_batch, p)
        datY_minibatch = datY[index, :].reshape(mini_batch, 1)
        grad_loss = gradient_loss(datX_minibatch, datY_minibatch, beta, mini_batch)
        ######################################
        #################
        ### update beta
        #################
        beta = beta - eta * grad_loss
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
        beta0 = beta0 - 0.001 * np.sum(datY_minibatch - beta0 * np.ones((mini_batch, 1)) - datX_minibatch.dot(beta)) / mini_batch
        ########################
    return(beta, beta0)
        
########################################################################################
np.random.seed(1000)
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
trainX, trainY, betastar_vec, istar = datageneration.generate_data(gen_data)
#RDA_para = {"lbd":150, "sigma":1000, "mini_batch":25}
OPG_para = {"lbd":270, "eta":0.001, "eta1":0.001, "mini_batch":25}
n_tr, p_tr = trainX.shape
beta = np.zeros((p_tr, 1))
beta0 = 0
#beta_RDA, beta0_RDA = RDA_elnet(trainX, trainY, beta, beta0, RDA_para)
beta_OPG, beta0_OPG = OPG_Lasso(trainX, trainY, beta, beta0, OPG_para)
print(np.flatnonzero(beta_OPG).shape)
print(beta0_OPG)
########################################################################################
#########
### Generate test data
#########
gen_testdata = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
testX, testY, betastar_vec, istar = datageneration.generate_data(gen_testdata)
########
### RMSE
########
testY_hat = testX.dot(beta_OPG) + beta0_OPG * np.ones((1000, 1))
err_hat = testY.T - testY_hat.T
rmse = np.sqrt(np.sum(err_hat**2) / 1000) 
print(rmse)
########################################################################################    