# -*- coding: utf-8 -*-
######################################################
### SADMM numerical experiment
######################################################
#############
### import package
#############
import numpy as np
import datageneration
import sys
sys.path.append("/Users/lizhesun/Documents/OFSelection_2023/simulations/regression")
import SADMM
import time
############
###############################################################################
def SADMM_numexp(gen_data, par, lbd_par, exp_times):
    n = gen_data["n"]
    p = gen_data["p"]
    k = gen_data["k"]
    alpha = gen_data["alpha"]
    beta_star = gen_data["beta_star"]
    dat_type = gen_data["dat_type"]
    rho = par["rho"]
    eta = par["eta"]
    mini_batch = par["mini_batch"]
    ##########################################
    if n >= p:
        batch = p
    else:
        batch = n
    num_batch = int(n / batch)
    result_mat = np.zeros((exp_times, 5))
    ##########################################
    ### Loop
    for i in range(exp_times):
        ###
        ### Generate seed
        ###
        np.random.seed(i + 100)
        result_mat[i,0] = i + 100
        ############################
        ###########
        ###lbd
        ###########
        lbd_start = lbd_par["lbd_start"]
        lbd_end = lbd_par["lbd_end"]
        lbd_vec = np.exp(np.linspace(lbd_start, lbd_end, 200))
        beta_mat = np.zeros((p, lbd_vec.shape[0]))
        theta_mat = np.zeros((p, lbd_vec.shape[0]))
        mu_mat = np.zeros((p, lbd_vec.shape[0]))
        intercept_vec = np.zeros(lbd_vec.shape[0])
        sel_fea_vec = np.zeros(lbd_vec.shape[0])
        #############################
        time_total = 0
        #################################
        gen_data_partial = {"n":batch, "p":p, "k":k, "alpha":alpha, "beta_star":beta_star, "dat_type":dat_type}
        #################################
        for j in range(num_batch):
            ### generate data
            Xtr_temp, Ytr_temp, betastar_vec, istar = datageneration.generate_data(gen_data_partial)
            #########
            time_inner = 0
            #########
            for Iter in range(lbd_vec.shape[0]):
                ################################
                lbd = lbd_vec[Iter]
                beta = beta_mat[:, Iter].reshape(p, 1)
                theta = theta_mat[:, Iter].reshape(p, 1)
                mu = mu_mat[:, Iter].reshape(p, 1)
                intercept = intercept_vec[Iter]
                #################################
                SADMM_para = {"lbd":lbd, "rho":rho, "eta":eta, "mini_batch":mini_batch}
                time_start = time.time()
                beta, theta, mu, intercept = SADMM.SADMM_OPG_Lasso(Xtr_temp, Ytr_temp, beta, theta, mu, intercept, SADMM_para)
                time_end = time.time()
                #################################
                sel_fea_vec[Iter] = np.flatnonzero(theta).shape[0]
                beta_mat[:, Iter] = beta.ravel()
                theta_mat[:, Iter] = theta.ravel()
                mu_mat[:, Iter] = mu.ravel()
                intercept_vec[Iter] = intercept
                ##################################
                time_cost = time_end - time_start
                time_inner = time_inner + time_cost
                ##################################
            time_total = time_total + time_inner
            ############################################################################
        result_mat[i, 4] = time_total
        lbd_index = np.where(sel_fea_vec <= k)[0]
        lbd_sel_index = lbd_index[0]
        lbd_select = lbd_vec[lbd_sel_index]
        result_mat[i, 1] = lbd_select
        print(sel_fea_vec[lbd_sel_index], i)
        theta_ADMM = theta_mat[:, lbd_sel_index].reshape(p, 1)
        intercept_ADMM = intercept_vec[lbd_sel_index]
        theta_index = np.flatnonzero(theta_ADMM)
        num_true_var = len(np.intersect1d(istar, theta_index))
        result_mat[i, 2] = num_true_var
        #########
        ### Generate test data
        #########
        gen_testdata = {"n":batch, "p":p, "k":k, "alpha":alpha, "beta_star":beta_star, "dat_type":dat_type}
        testX, testY, betastar_vec, istar = datageneration.generate_data(gen_testdata)
        ########
        ### RMSE
        ########
        testY_hat = testX.dot(theta_ADMM) + intercept_ADMM * np.ones((batch, 1))
        err_hat = testY.T - testY_hat.T
        rmse = np.sqrt(np.sum(err_hat**2) / batch) 
        result_mat[i, 3] = rmse
        ########    
    return(result_mat)
###############################################################################
###
##############################################################################
###beta = 1
##############################################################################
###
gen_data = {"n":300, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
lbd_par = {"lbd_start":-1, "lbd_end":1}
par = {"rho":1, "eta":0.0001, "mini_batch":25}
SADMM_300_beta1_mat = SADMM_numexp(gen_data, par, lbd_par, 100)
DR_exp00 = np.sum(SADMM_300_beta1_mat[:,2] == gen_data["k"])
PCD_exp00 = np.mean(SADMM_300_beta1_mat[:,2] / gen_data["k"])
rmse_ave_exp00 = np.mean(SADMM_300_beta1_mat[:,3])
running_time_ave_exp00 = np.mean(SADMM_300_beta1_mat[:,4])
print(DR_exp00)
print(PCD_exp00)
print(rmse_ave_exp00)
print(running_time_ave_exp00)
###
###############################################################################
###
gen_data = {"n":500, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
lbd_par = {"lbd_start":-1, "lbd_end":1}
par = {"rho":1, "eta":0.0001, "mini_batch":25}
SADMM_500_beta1_mat = SADMM_numexp(gen_data, par, lbd_par, 100)
DR_exp0 = np.sum(SADMM_500_beta1_mat[:,2] == gen_data["k"])
PCD_exp0 = np.mean(SADMM_500_beta1_mat[:,2] / gen_data["k"])
rmse_ave_exp0 = np.mean(SADMM_500_beta1_mat[:,3])
running_time_ave_exp0 = np.mean(SADMM_500_beta1_mat[:,4])
print(DR_exp0)
print(PCD_exp0)
print(rmse_ave_exp0)
print(running_time_ave_exp0)
###
###############################################################################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
lbd_par = {"lbd_start":0.5, "lbd_end":1.8}
par = {"rho":1, "eta":0.0001, "mini_batch":25}
SADMM_1000_beta1_mat = SADMM_numexp(gen_data, par, lbd_par, 100)
DR_exp1 = np.sum(SADMM_1000_beta1_mat[:,2] == gen_data["k"])
PCD_exp1 = np.mean(SADMM_1000_beta1_mat[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(SADMM_1000_beta1_mat[:,3])
running_time_ave_exp1 = np.mean(SADMM_1000_beta1_mat[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
###############################################################################
###
gen_data = {"n":3000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
lbd_par = {"lbd_start":1.2, "lbd_end":2.8}
par = {"rho":1, "eta":0.0001, "mini_batch":25}
SADMM_3000_beta1_mat = SADMM_numexp(gen_data, par, lbd_par, 100)
DR_exp2 = np.sum(SADMM_3000_beta1_mat[:,2] == gen_data["k"])
PCD_exp2 = np.mean(SADMM_3000_beta1_mat[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(SADMM_3000_beta1_mat[:,3])
running_time_ave_exp2 = np.mean(SADMM_3000_beta1_mat[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
###############################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
lbd_par = {"lbd_start":2.2, "lbd_end":3.7}
par = {"rho":1, "eta":0.0001, "mini_batch":25}
SADMM_10000_beta1_mat = SADMM_numexp(gen_data, par, lbd_par, 100)
DR_exp3 = np.sum(SADMM_10000_beta1_mat[:,2] == gen_data["k"])
PCD_exp3 = np.mean(SADMM_10000_beta1_mat[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(SADMM_10000_beta1_mat[:,3])
running_time_ave_exp3 = np.mean(SADMM_10000_beta1_mat[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
###############################################################################
###
##############################################################################
###beta = 0.1
##############################################################################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":0.1, "dat_type":1}
lbd_par = {"lbd_start":-1.6, "lbd_end":0}
par = {"rho":1, "eta":0.0001, "mini_batch":25}
SADMM_1000_beta01_mat = SADMM_numexp(gen_data, par, lbd_par, 100)
DR_exp1 = np.sum(SADMM_1000_beta01_mat[:,2] == gen_data["k"])
PCD_exp1 = np.mean(SADMM_1000_beta01_mat[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(SADMM_1000_beta01_mat[:,3])
running_time_ave_exp1 = np.mean(SADMM_1000_beta01_mat[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
###############################################################################
###
gen_data = {"n":3000, "p":1000, "k":100, "alpha":1, "beta_star":0.1, "dat_type":1}
lbd_par = {"lbd_start":-0.6, "lbd_end":1}
par = {"rho":1, "eta":0.0001, "mini_batch":25}
SADMM_3000_beta01_mat = SADMM_numexp(gen_data, par, lbd_par, 100)
DR_exp2 = np.sum(SADMM_3000_beta01_mat[:,2] == gen_data["k"])
PCD_exp2 = np.mean(SADMM_3000_beta01_mat[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(SADMM_3000_beta01_mat[:,3])
running_time_ave_exp2 = np.mean(SADMM_3000_beta01_mat[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
###############################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":0.1, "dat_type":1}
lbd_par = {"lbd_start":0.4, "lbd_end":2}
par = {"rho":1, "eta":0.0001, "mini_batch":25}
SADMM_10000_beta01_mat = SADMM_numexp(gen_data, par, lbd_par, 100)
DR_exp3 = np.sum(SADMM_10000_beta01_mat[:,2] == gen_data["k"])
PCD_exp3 = np.mean(SADMM_10000_beta01_mat[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(SADMM_10000_beta01_mat[:,3])
running_time_ave_exp3 = np.mean(SADMM_10000_beta01_mat[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
###############################################################################
###
###############################################################################
###beta = 0.01
###############################################################################
###
gen_data = {"n":500, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_par = {"lbd_start":-5, "lbd_end":-3}
par = {"rho":1, "eta":0.0001, "mini_batch":25}
SADMM_500_beta001_mat = SADMM_numexp(gen_data, par, lbd_par, 100)
DR_exp0 = np.sum(SADMM_500_beta001_mat[:,2] == gen_data["k"])
PCD_exp0 = np.mean(SADMM_500_beta001_mat[:,2] / gen_data["k"])
rmse_ave_exp0 = np.mean(SADMM_500_beta001_mat[:,3])
running_time_ave_exp0 = np.mean(SADMM_500_beta001_mat[:,4])
print(DR_exp0)
print(PCD_exp0)
print(rmse_ave_exp0)
print(running_time_ave_exp0)
###
###############################################################################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_par = {"lbd_start":-4.2, "lbd_end":-2}
par = {"rho":1, "eta":0.0001, "mini_batch":25}
SADMM_1000_beta001_mat = SADMM_numexp(gen_data, par, lbd_par, 100)
DR_exp1 = np.sum(SADMM_1000_beta001_mat[:,2] == gen_data["k"])
PCD_exp1 = np.mean(SADMM_1000_beta001_mat[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(SADMM_1000_beta001_mat[:,3])
running_time_ave_exp1 = np.mean(SADMM_1000_beta001_mat[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
###############################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_par = {"lbd_start":-2, "lbd_end":1}
par = {"rho":1, "eta":0.0001, "mini_batch":25}
SADMM_10000_beta001_mat = SADMM_numexp(gen_data, par, lbd_par, 100)
DR_exp2 = np.sum(SADMM_10000_beta001_mat[:,2] == gen_data["k"])
PCD_exp2 = np.mean(SADMM_10000_beta001_mat[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(SADMM_10000_beta001_mat[:,3])
running_time_ave_exp2 = np.mean(SADMM_10000_beta001_mat[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
###############################################################################
###
gen_data = {"n":100000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_par = {"lbd_start":-1, "lbd_end":2}
par = {"rho":1, "eta":0.0001, "mini_batch":25}
SADMM_100000_beta001_mat = SADMM_numexp(gen_data, par, lbd_par, 100)
DR_exp3 = np.sum(SADMM_100000_beta001_mat[:,2] == gen_data["k"])
PCD_exp3 = np.mean(SADMM_100000_beta001_mat[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(SADMM_100000_beta001_mat[:,3])
running_time_ave_exp3 = np.mean(SADMM_100000_beta001_mat[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
###############################################################################
###
gen_data = {"n":300000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_par = {"lbd_start":0, "lbd_end":3}
par = {"rho":1, "eta":0.0001, "mini_batch":25}
SADMM_300000_beta001_mat = SADMM_numexp(gen_data, par, lbd_par, 100)
DR_exp4 = np.sum(SADMM_300000_beta001_mat[:,2] == gen_data["k"])
PCD_exp4 = np.mean(SADMM_300000_beta001_mat[:,2] / gen_data["k"])
rmse_ave_exp4 = np.mean(SADMM_300000_beta001_mat[:,3])
running_time_ave_exp4 = np.mean(SADMM_300000_beta001_mat[:,4])
print(DR_exp4)
print(PCD_exp4)
print(rmse_ave_exp4)
print(running_time_ave_exp4)
###
###############################################################################
###
gen_data = {"n":1000000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_par = {"lbd_start":0, "lbd_end":3}
par = {"rho":1, "eta":0.0001, "mini_batch":25}
SADMM_1000000_beta001_mat = SADMM_numexp(gen_data, par, lbd_par, 100)
DR_exp5 = np.sum(SADMM_1000000_beta001_mat[:,2] == gen_data["k"])
PCD_exp5 = np.mean(SADMM_1000000_beta001_mat[:,2] / gen_data["k"])
rmse_ave_exp5 = np.mean(SADMM_1000000_beta001_mat[:,3])
running_time_ave_exp5 = np.mean(SADMM_1000000_beta001_mat[:,4])
print(DR_exp5)
print(PCD_exp5)
print(rmse_ave_exp5)
print(running_time_ave_exp5)
###
###############################################################################
###
### Big data experiment
###
###############################################################################
###beta = 1
###############################################################################
###
gen_data = {"n":10000, "p":10000, "k":1000, "alpha":1, "beta_star":1, "dat_type":1}
lbd_par = {"lbd_start":3.5, "lbd_end":7}
par = {"rho":1, "eta":0.0001, "mini_batch":25}
SADMM_10000_beta1_mat = SADMM_numexp(gen_data, par, lbd_par, 20)
DR_exp1 = np.sum(SADMM_10000_beta1_mat[:,2] == gen_data["k"])
PCD_exp1 = np.mean(SADMM_10000_beta1_mat[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(SADMM_10000_beta1_mat[:,3])
running_time_ave_exp1 = np.mean(SADMM_10000_beta1_mat[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
###############################################################################
###
gen_data = {"n":30000, "p":10000, "k":1000, "alpha":1, "beta_star":1, "dat_type":1}
lbd_par = {"lbd_start":2.5, "lbd_end":6.5}
par = {"rho":1, "eta":0.0001, "mini_batch":25}
SADMM_30000_beta1_mat = SADMM_numexp(gen_data, par, lbd_par, 20)
DR_exp2 = np.sum(SADMM_30000_beta1_mat[:,2] == gen_data["k"])
PCD_exp2 = np.mean(SADMM_30000_beta1_mat[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(SADMM_30000_beta1_mat[:,3])
running_time_ave_exp2 = np.mean(SADMM_30000_beta1_mat[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
###############################################################################
###
gen_data = {"n":100000, "p":10000, "k":1000, "alpha":1, "beta_star":1, "dat_type":1}
lbd_par = {"lbd_start":3.6, "lbd_end":6.2}
par = {"rho":1, "eta":0.0001, "mini_batch":25}
SADMM_100000_beta1_mat = SADMM_numexp(gen_data, par, lbd_par, 20)
DR_exp3 = np.sum(SADMM_100000_beta1_mat[:,2] == gen_data["k"])
PCD_exp3 = np.mean(SADMM_100000_beta1_mat[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(SADMM_100000_beta1_mat[:,3])
running_time_ave_exp3 = np.mean(SADMM_100000_beta1_mat[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
###############################################################################
###
##############################
###beta = 0.1
##############################
gen_data = {"n":10000, "p":10000, "k":1000, "alpha":1, "beta_star":0.1, "dat_type":1}
lbd_par = {"lbd_start":-1, "lbd_end":4}
par = {"rho":1, "eta":0.0001, "mini_batch":25}
SADMM_10000_beta01_mat = SADMM_numexp(gen_data, par, lbd_par, 20)
DR_exp1 = np.sum(SADMM_10000_beta01_mat[:,2] == gen_data["k"])
PCD_exp1 = np.mean(SADMM_10000_beta01_mat[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(SADMM_10000_beta01_mat[:,3])
running_time_ave_exp1 = np.mean(SADMM_10000_beta01_mat[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
###############################################################################
###
gen_data = {"n":30000, "p":10000, "k":1000, "alpha":1, "beta_star":0.1, "dat_type":1}
lbd_par = {"lbd_start":-1, "lbd_end":4}
par = {"rho":1, "eta":0.0001, "mini_batch":25}
SADMM_30000_beta01_mat = SADMM_numexp(gen_data, par, lbd_par, 20)
DR_exp2 = np.sum(SADMM_30000_beta01_mat[:,2] == gen_data["k"])
PCD_exp2 = np.mean(SADMM_30000_beta01_mat[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(SADMM_30000_beta01_mat[:,3])
running_time_ave_exp2 = np.mean(SADMM_30000_beta01_mat[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
###############################################################################
###
gen_data = {"n":100000, "p":10000, "k":1000, "alpha":1, "beta_star":0.1, "dat_type":1}
lbd_par = {"lbd_start":0, "lbd_end":4}
par = {"rho":1, "eta":0.0001, "mini_batch":25}
SADMM_100000_beta01_mat = SADMM_numexp(gen_data, par, lbd_par, 20)
DR_exp3 = np.sum(SADMM_100000_beta01_mat[:,2] == gen_data["k"])
PCD_exp3 = np.mean(SADMM_100000_beta01_mat[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(SADMM_100000_beta01_mat[:,3])
running_time_ave_exp3 = np.mean(SADMM_100000_beta01_mat[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
###############################################################################
###beta = 0.01
###############################################################################
###
gen_data = {"n":10000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_par = {"lbd_start":-2, "lbd_end":1.8}
par = {"rho":1, "eta":0.0001, "mini_batch":25}
SADMM_10000_beta001_mat = SADMM_numexp(gen_data, par, lbd_par, 20)
DR_exp1 = np.sum(SADMM_10000_beta001_mat[:,2] == gen_data["k"])
PCD_exp1 = np.mean(SADMM_10000_beta001_mat[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(SADMM_10000_beta001_mat[:,3])
running_time_ave_exp1 = np.mean(SADMM_10000_beta001_mat[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
###############################################################################
###
gen_data = {"n":30000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_par = {"lbd_start":-1.5, "lbd_end":2.2}
par = {"rho":1, "eta":0.0001, "mini_batch":25}
SADMM_30000_beta001_mat = SADMM_numexp(gen_data, par, lbd_par, 20)
DR_exp2 = np.sum(SADMM_30000_beta001_mat[:,2] == gen_data["k"])
PCD_exp2 = np.mean(SADMM_30000_beta001_mat[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(SADMM_30000_beta001_mat[:,3])
running_time_ave_exp2 = np.mean(SADMM_30000_beta001_mat[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
###############################################################################
###
gen_data = {"n":100000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_par = {"lbd_start":-1, "lbd_end":2.4}
par = {"rho":1, "eta":0.0001, "mini_batch":25}
SADMM_100000_beta001_mat = SADMM_numexp(gen_data, par, lbd_par, 20)
DR_exp3 = np.sum(SADMM_100000_beta001_mat[:,2] == gen_data["k"])
PCD_exp3 = np.mean(SADMM_100000_beta001_mat[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(SADMM_100000_beta001_mat[:,3])
running_time_ave_exp3 = np.mean(SADMM_100000_beta001_mat[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
###############################################################################
###
gen_data = {"n":300000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_par = {"lbd_start":-0.8, "lbd_end":2.6}
par = {"rho":1, "eta":0.0001, "mini_batch":25}
SADMM_300000_beta001_mat = SADMM_numexp(gen_data, par, lbd_par, 20)
DR_exp4 = np.sum(SADMM_300000_beta001_mat[:,2] == gen_data["k"])
PCD_exp4 = np.mean(SADMM_300000_beta001_mat[:,2] / gen_data["k"])
rmse_ave_exp4 = np.mean(SADMM_300000_beta001_mat[:,3])
running_time_ave_exp4 = np.mean(SADMM_300000_beta001_mat[:,4])
print(DR_exp4)
print(PCD_exp4)
print(rmse_ave_exp4)
print(running_time_ave_exp4)
###
###############################################################################
####
#gen_data = {"n":1000000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
#lbd_par = {"lbd_start":0.5, "lbd_end":6}
#par = {"rho":1, "eta":0.0001, "mini_batch":25}
#SADMM_1000000_beta001_mat = SADMM_numexp(gen_data, par, lbd_par, 10)
#DR_exp5 = np.sum(SADMM_1000000_beta001_mat[:,2] == gen_data["k"])
#PCD_exp5 = np.mean(SADMM_1000000_beta001_mat[:,2] / gen_data["k"])
#rmse_ave_exp5 = np.mean(SADMM_1000000_beta001_mat[:,3])
#running_time_ave_exp5 = np.mean(SADMM_1000000_beta001_mat[:,4])
#print(DR_exp5)
#print(PCD_exp5)
#print(rmse_ave_exp5)
#print(running_time_ave_exp5)

























   
