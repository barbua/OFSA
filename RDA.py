# -*- coding: utf-8 -*-
####################################
### Lizhe Sun  
### Dual Averaging Algorithms
####################################
##################
##Load Package
##################
import datageneration
import numpy as np
##################
##################
### one step RDA (Regression)
##################
def gradient_loss(X, Y, beta, mini_batch):
    temp1 = Y - X.dot(beta)
    gradloss = - 2 * X.T.dot(temp1)
    gradloss = gradloss / mini_batch
    return gradloss
###
########################################################################################
###
def RDA_elnet(datX, datY, beta, beta0, RDA_para):
    n, p = datX.shape
    lbd = RDA_para["lbd"]
    sigma = RDA_para["sigma"]
    mini_batch = RDA_para["mini_batch"]
    N_iter = int(n / mini_batch)
    ave_grad = np.zeros((p, 1))
    ###################################
    for i in range(N_iter):
        index = np.arange(i * mini_batch, (i+1) * mini_batch, 1)
        datX_minibatch = datX[index, :].reshape(mini_batch, p)
        datY_minibatch = datY[index, :].reshape(mini_batch, 1)
        grad_loss = gradient_loss(datX_minibatch, datY_minibatch, beta, mini_batch)
        ave_grad = (i / (i+1)) * ave_grad + (1 / (i+1)) * grad_loss
        #######################
        ### update beta
        #######################
        #######################
        for j in range(p):
            if ave_grad[j, 0] > lbd:
                beta[j, 0] = - (ave_grad[j, 0] - lbd) / sigma
            elif ave_grad[j, 0] < -lbd:
                beta[j, 0] = - (ave_grad[j, 0] + lbd) / sigma
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
###############
### RDA_lasso
###############
def RDA_lasso(datX, datY, beta, beta0, RDA_para):
    n, p = datX.shape
    lbd = RDA_para["lbd"]
    gamma = RDA_para["gamma"]
    rho = RDA_para["rho"]
    mini_batch = RDA_para["mini_batch"]
    ###################################
    N_iter = int(n / mini_batch)
    ave_grad = np.zeros((p, 1))
    ###################################
    for t in range(N_iter):
        index = np.arange(t * mini_batch, (t+1) * mini_batch)
        datX_minibatch = datX[index, :].reshape(mini_batch, p)
        datY_minibatch = datY[index,:].reshape(mini_batch, 1)
        grad_loss = gradient_loss(datX_minibatch, datY_minibatch, beta, mini_batch)
        ave_grad = (t / (t+1)) * ave_grad + (1 / (t+1)) * grad_loss
        #######################
        ### update beta
        #######################
        lbd_RDA = lbd + gamma * rho / np.sqrt(t+1)
        #######################
        for j in range(p):
            if ave_grad[j, 0] > lbd_RDA:
                beta[j, 0] = - np.sqrt(t+1) * (ave_grad[j, 0] - lbd_RDA) / gamma
            elif ave_grad[j, 0] < - lbd_RDA:
                beta[j, 0] = - np.sqrt(t+1) * (ave_grad[j, 0] + lbd_RDA) / gamma
            else:
                beta[j, 0] = 0
        ########################
        ########################
        ### update beta0
        ########################
        beta0 = beta0 - 0.001 * np.sum(datY_minibatch - beta0 * np.ones((mini_batch, 1)) - datX_minibatch.dot(beta)) / mini_batch
        ########################
    return(beta, beta0)
        
    
        
    
#########################################################################################
#np.random.seed(1000)
#gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
#trainX, trainY, betastar_vec, istar = datageneration.generate_data(gen_data)
##RDA_para = {"lbd":150, "sigma":1000, "mini_batch":25}
#RDA_para = {"lbd":198, "gamma":5000, "rho":0.005, "mini_batch":25}
#n_tr, p_tr = trainX.shape
#beta = np.zeros((p_tr, 1))
#beta0 = 0
##beta_RDA, beta0_RDA = RDA_elnet(trainX, trainY, beta, beta0, RDA_para)
#beta_RDA, beta0_RDA = RDA_lasso(trainX, trainY, beta, beta0, RDA_para)
#print(np.flatnonzero(beta_RDA).shape)
#print(beta0_RDA)
#########################################################################################
##########
#### Generate test data
##########
#gen_testdata = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
#testX, testY, betastar_vec, istar = datageneration.generate_data(gen_testdata)
#########
#### RMSE
#########
#testY_hat = testX.dot(beta_RDA) + beta0_RDA * np.ones((1000, 1))
#err_hat = testY.T - testY_hat.T
#rmse = np.sqrt(np.sum(err_hat**2) / 1000) 
#print(rmse)
#########################################################################################
#


    