# -*- coding: utf-8 -*-
######################################################
### SADMM Algorithm
######################################################
#############
### import package
#############
import numpy as np
#import datageneration
############
############
### ADMM for Lasso (Batch Algorithm)
############
### beta0, theta0, mu0, lbd, rho, eta
############
def OADMM_Lasso(datX, datY, beta0, theta0, mu0, intercep, OADMM_para):
    n, p = datX.shape
    lbd = OADMM_para["lbd"]
    rho = OADMM_para["rho"]
    eta = OADMM_para["eta"]
    thr = lbd / rho
    #######################
    ### update beta theta mu
    #######################
    for i in range(n):
        x_obs = datX[i, :].reshape(p, 1)
        y_obs = datY[i, 0]
        ######################
        xx_obs = x_obs.T.dot(x_obs)
        xx_temp = np.eye(p) - 1 / (eta + rho + xx_obs) * x_obs.dot(x_obs.T)
        beta = xx_temp.dot(y_obs * x_obs + rho * (theta0 - mu0) + eta * beta0)
        beta = beta / (eta + rho)
        ########################
        theta = beta + mu0
        for j in range(p):
            if theta[j, 0] > thr:
                theta[j, 0] = theta[j, 0] - thr
            elif theta[j, 0] < - thr:
                theta[j, 0] = theta[j, 0] + thr
            else:
                theta[j, 0] = 0
        ##########################
        intercep = intercep + 0.0001 * (y_obs - intercep - x_obs.T.dot(theta)[0, 0])
        ##########################
        mu = mu0 + beta - theta
        ##########################
        beta0 = beta
        theta0 = theta
        mu0 = mu
    return (beta, theta, mu, intercep)
########################################################################################
##################
###SADMM_OPG
##################
########################################################################################
def SADMM_OPG_Lasso(datX, datY, beta, theta, mu, intercept, SADMM_para):
    n, p = datX.shape
    lbd = SADMM_para["lbd"]
    rho = SADMM_para["rho"]
    eta = SADMM_para["eta"]
    mini_batch = SADMM_para["mini_batch"]
    thr = lbd / rho
    Iter_time = int(n / mini_batch)
    ###############################
    #beta_ave = beta0
    #theta_ave = theta0
    ###################
    ### update beta theta mu
    ###################
    for i in range(Iter_time):
        index = np.arange(mini_batch * i, mini_batch * (i+1), 1)
        datX_mb = datX[index, :].reshape(mini_batch, p)
        datY_mb = datY[index, :].reshape(mini_batch, 1)
        grad_mb = - datX_mb.T.dot(datY_mb - intercept * np.ones((mini_batch, 1)) - datX_mb.dot(beta))
        grad_mb = grad_mb / mini_batch
        beta = - grad_mb + rho * (theta + mu) + beta / eta
        beta = beta / (rho + 1 / eta)
        ###########################
        theta = beta - mu
        for j in range(p):
            if theta[j, 0] > thr:
                theta[j, 0] = theta[j, 0] - thr
            elif theta[j, 0] < - thr:
                theta[j, 0] = theta[j, 0] + thr
            else:
                theta[j, 0] = 0
        ###########################
        mu = mu - (beta - theta)
        ###########################
        intercept = intercept - 0.0001 * np.sum(datY_mb - intercept * np.ones((mini_batch, 1)) - datX_mb.dot(theta)) / mini_batch
        ############################
        ###########################
    return (beta, theta, mu, intercept)
########################################################################################
########################
#### SADMM_RDA_Lasso
########################
#########################################################################################
#np.random.seed(1001)
#gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
#trainX, trainY, betastar_vec, istar = datageneration.generate_data(gen_data)
##OADMM_para = {"lbd":37, "rho":1, "eta":1000}
#SADMM_para = {"lbd":35.5, "rho":1, "eta":0.001, "mini_batch":25}
#n_tr, p_tr = trainX.shape
#beta0 = np.zeros((p_tr, 1))
#theta0 = np.zeros((p_tr, 1))
#mu0 = np.zeros((p_tr, 1))
#intercept0 = 0
##iterated_times = 0
###beta_est, theta_est, mu_est, intercep = OADMM_Lasso(trainX, trainY, beta0, theta0, mu0, intercep, OADMM_para)
#beta_est, theta_est, mu_est, intercept = SADMM_OPG_Lasso(trainX, trainY, beta0, theta0, mu0, intercept0, SADMM_para)
#print(np.flatnonzero(theta_est).shape)
#print(intercept)
##########################################################################################
##########################################################################################
###########
##### Generate test data
###########
#gen_testdata = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
#testX, testY, betastar_vec, istar = datageneration.generate_data(gen_testdata)
##########
##### RMSE
##########
#testY_hat = testX.dot(theta_est) + intercept * np.ones((gen_testdata["n"], 1))
#err_hat = testY.T - testY_hat.T
#l2_loss = np.sum(err_hat**2)
#print(l2_loss)
##########################################################################################



