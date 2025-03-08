# -*- coding: utf-8 -*-
######################################
###  online FSA numerical experiment
###
######################################
###         Load package
######################################
import onlineFSA
import time
import sys
sys.path.append("/Users/lizhesun/Documents/OFSelection_2023/simulations/regression")
import numpy as np
import datageneration
########################################################################################
###
########################################################################################
###  onlineFSA data experiment
########################################################################################
###
### exp_para = {"n":n, "p":p, "k":k, "alpha":alpha, "beta_star":beta_star, "dat_type":dat_type, "eta":}
def onlineFSA_numexp(gen_data, eta, lbd, loop_time):
    ###
    ## Parameters
    ###
    n = gen_data["n"]
    p = gen_data["p"]
    k = gen_data["k"]
    alpha = gen_data["alpha"]
    beta_star = gen_data["beta_star"]
    dat_type = gen_data["dat_type"]
    result_mat = np.zeros((loop_time, 5))
    ### Batch for data
    if p > n:
        batch = n
    else:
        batch = p
    ################
    num_batch = int(n / batch)
    ### Looping
    for i in range(loop_time):
        ###set seed
        np.random.seed(i+100)
        ## Store seed
        result_mat[i, 0] = i + 100
        result_mat[i, 1] = eta
        ########
        ### Generate data
        ### initial value for runningsums
        ########
        n_sum = 0
        Sx_sum = np.zeros((1,p))
        Sy_sum = 0
        Sxx_sum = np.zeros((p,p))
        Sxy_sum = np.zeros((p,1))
        Syy_sum = 0
        rs_sum = {"n":n_sum, "Sx":Sx_sum, "Sy":Sy_sum, "Sxx":Sxx_sum, "Sxy":Sxy_sum, "Syy":Syy_sum}
        gen_data_partial = {"n":batch, "p":p, "k":k, "alpha":alpha, "beta_star":beta_star, "dat_type":dat_type}
        for j in range(num_batch):
            temp_trainX, temp_trainY, betastar_vec, istar = datageneration.generate_data(gen_data_partial)
            rs_temp = onlineFSA.running_aves(temp_trainX, temp_trainY)
            rs_sum = onlineFSA.add_runningaves(rs_sum, rs_temp)
        ####
        ####
        del temp_trainX, temp_trainY
        ####
        if rs_sum["n"] != n:
            print("error")
            break
        ###
        ## Standardize Running Sum
        standardize_runningsum = rs_sum
        XX_normalize, XY_normalize, mu_x, mu_y, std_x = onlineFSA.standardize_ra(standardize_runningsum)
        ##
        ##
        FSA_para = {"n":n, "k":k, "eta":eta, "mu":10, "lbd":lbd, "N_iter":200}
        ## Setup start time
        t_start = time.process_time()
        ## FSA Experiment
        FSA_experiment = onlineFSA.onlineFSA(XX_normalize, XY_normalize, FSA_para, 15)
        ## end time
        t_end = time.process_time()
        ## Calculate the running time 
        running_time = t_end - t_start
        result_mat[i, 4] = running_time
        ## Compare the index with the true index
        FSA_sel = FSA_experiment[1]
        num_true_var = len(np.intersect1d(istar, FSA_sel))
        result_mat[i, 2] = num_true_var
        ## RMSE
        ## Generate Test Data
        gen_testdata = {"n":batch, "p":p, "k":k, "alpha":alpha, "beta_star":beta_star, "dat_type":dat_type}
        testX, testY, betastar_vec, istar = datageneration.generate_data(gen_testdata)
        ################
        ### Standardize the test data
        ################
        testY_center = testY - mu_y * np.ones((batch, 1))
        testX_standardize = testX - np.ones((batch, 1)).dot(mu_x)
        inv_sigma = 1 / std_x
        testX_standardize = inv_sigma * testX_standardize
        #####
        ### refit by using OLS
        #####
        XX_sel = XX_normalize[np.ix_(FSA_sel, FSA_sel)]
        XY_sel = XY_normalize[FSA_sel]
        beta_ols = onlineFSA.OLS_runningaves(XX_sel, XY_sel, 0)
        beta_hat = np.zeros((p,1))
        beta_hat[FSA_sel] = beta_ols
        ######
        #####
        ######
        testY_hat = testX_standardize.dot(beta_hat)
        err_hat = testY_center.T - testY_hat.T
        rmse = np.sqrt(np.sum(err_hat**2) / batch) 
        result_mat[i, 3] = rmse
        #print(i)
    return(result_mat)
###################################################################################
#### n = 300, 1000, 3000, 10000, 30000, 100000, p = 1000, k = 100, alpha = 1, beta = 1, 0.1, 0.01 
###################################################################################
#######################
### beta = 1
#######################
gen_data = {"n":300, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
FSA_300_mat = onlineFSA_numexp(gen_data, 0.01, 0, 100)
DR_exp0 = np.sum(FSA_300_mat[:,2] == gen_data["k"])
PCD_exp0 = np.mean(FSA_300_mat[:,2] / gen_data["k"])
rmse_ave_exp0 = np.mean(FSA_300_mat[:,3])
running_time_ave_exp0 = np.mean(FSA_300_mat[:,4])
print(DR_exp0)
print(PCD_exp0)
print(rmse_ave_exp0)
print(running_time_ave_exp0)
###
########################################################################################
###
gen_data = {"n":500, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
FSA_500_mat = onlineFSA_numexp(gen_data, 0.01, 0, 100)
DR_exp1 = np.sum(FSA_500_mat[:,2] == gen_data["k"])
PCD_exp1 = np.mean(FSA_500_mat[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(FSA_500_mat[:,3])
running_time_ave_exp1 = np.mean(FSA_500_mat[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
########################################################################################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
FSA_1000_mat = onlineFSA_numexp(gen_data, 0.01, 0, 100)
DR_exp2 = np.sum(FSA_1000_mat[:,2] == gen_data["k"])
PCD_exp2 = np.mean(FSA_1000_mat[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(FSA_1000_mat[:,3])
running_time_ave_exp2 = np.mean(FSA_1000_mat[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
##################################################################################
###
gen_data = {"n":3000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
FSA_3000_mat = onlineFSA_numexp(gen_data, 0.01, 0, 100)
DR_exp3 = np.sum(FSA_3000_mat[:,2] == gen_data["k"])
PCD_exp3 = np.mean(FSA_3000_mat[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(FSA_3000_mat[:,3])
running_time_ave_exp3 = np.mean(FSA_3000_mat[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
####
##################################################################################
####
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
FSA_10000_mat = onlineFSA_numexp(gen_data, 0.01, 0, 100)
DR_exp4 = np.sum(FSA_10000_mat[:,2] == gen_data["k"])
PCD_exp4 = np.mean(FSA_10000_mat[:,2] / gen_data["k"])
rmse_ave_exp4 = np.mean(FSA_10000_mat[:,3])
running_time_ave_exp4 = np.mean(FSA_10000_mat[:,4])
print(DR_exp4)
print(PCD_exp4)
print(rmse_ave_exp4)
print(running_time_ave_exp4)
###
##################################################################################
###
##################################################################################
### beta = 0.1
##################################################################################
###
#gen_data = {"n":300, "p":1000, "k":100, "alpha":1, "beta_star":0.1, "dat_type":1}
#FSA_300_mat = onlineFSA_numexp(gen_data, 0.01, 0, 100)
#DR_exp1 = np.sum(FSA_300_mat[:,2] == gen_data["k"])
#PCD_exp1 = np.mean(FSA_300_mat[:,2] / gen_data["k"])
#rmse_ave_exp1 = np.mean(FSA_300_mat[:,3])
#running_time_ave_exp1 = np.mean(FSA_300_mat[:,4])
#print(DR_exp1)
#print(PCD_exp1)
#print(rmse_ave_exp1)
#print(running_time_ave_exp1)
###
########################################################################################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":0.1, "dat_type":1}
FSA_1000_mat = onlineFSA_numexp(gen_data, 0.01, 0, 100)
DR_exp2 = np.sum(FSA_1000_mat[:,2] == gen_data["k"])
PCD_exp2 = np.mean(FSA_1000_mat[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(FSA_1000_mat[:,3])
running_time_ave_exp2 = np.mean(FSA_1000_mat[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
########################################################################################
###
gen_data = {"n":3000, "p":1000, "k":100, "alpha":1, "beta_star":0.1, "dat_type":1}
FSA_3000_mat = onlineFSA_numexp(gen_data, 0.01, 0, 100)
DR_exp3 = np.sum(FSA_3000_mat[:,2] == gen_data["k"])
PCD_exp3 = np.mean(FSA_3000_mat[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(FSA_3000_mat[:,3])
running_time_ave_exp3 = np.mean(FSA_3000_mat[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
########################################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":0.1, "dat_type":1}
FSA_10000_mat = onlineFSA_numexp(gen_data, 0.01, 0, 100)
DR_exp4 = np.sum(FSA_10000_mat[:,2] == gen_data["k"])
PCD_exp4 = np.mean(FSA_10000_mat[:,2] / gen_data["k"])
rmse_ave_exp4 = np.mean(FSA_10000_mat[:,3])
running_time_ave_exp4 = np.mean(FSA_10000_mat[:,4])
print(DR_exp4)
print(PCD_exp4)
print(rmse_ave_exp4)
print(running_time_ave_exp4)
###
########################################################################################
####
####
################################################################################
### Beta = 0.01
################################################################################
###
gen_data = {"n":500, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
FSA_500_mat = onlineFSA_numexp(gen_data, 0.01, 0, 100)
DR_exp0 = np.sum(FSA_500_mat[:,2] == gen_data["k"])
PCD_exp0 = np.mean(FSA_500_mat[:,2] / gen_data["k"])
rmse_ave_exp0 = np.mean(FSA_500_mat[:,3])
running_time_ave_exp0 = np.mean(FSA_500_mat[:,4])
print(DR_exp0)
print(PCD_exp0)
print(rmse_ave_exp0)
print(running_time_ave_exp0)
###
#################################################################################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
FSA_1000_mat = onlineFSA_numexp(gen_data, 0.01, 0, 100)
DR_exp1 = np.sum(FSA_1000_mat[:,2] == gen_data["k"])
PCD_exp1 = np.mean(FSA_1000_mat[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(FSA_1000_mat[:,3])
running_time_ave_exp1 = np.mean(FSA_1000_mat[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
#################################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
FSA_10000_mat = onlineFSA_numexp(gen_data, 0.01, 0, 100)
DR_exp2 = np.sum(FSA_10000_mat[:,2] == gen_data["k"])
PCD_exp2 = np.mean(FSA_10000_mat[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(FSA_10000_mat[:,3])
running_time_ave_exp2 = np.mean(FSA_10000_mat[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
#########################################################################################
###
gen_data = {"n":100000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
FSA_100000_mat = onlineFSA_numexp(gen_data, 0.01, 0, 100)
DR_exp3 = np.sum(FSA_100000_mat[:,2] == gen_data["k"])
PCD_exp3 = np.mean(FSA_100000_mat[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(FSA_100000_mat[:,3])
running_time_ave_exp3 = np.mean(FSA_100000_mat[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)     
###
########################################################################################
###
gen_data = {"n":300000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
FSA_300000_mat = onlineFSA_numexp(gen_data, 0.01, 0, 100)
DR_exp4 = np.sum(FSA_300000_mat[:,2] == gen_data["k"])
PCD_exp4 = np.mean(FSA_300000_mat[:,2] / gen_data["k"])
rmse_ave_exp4 = np.mean(FSA_300000_mat[:,3])
running_time_ave_exp4 = np.mean(FSA_300000_mat[:,4])
print(DR_exp4)
print(PCD_exp4)
print(rmse_ave_exp4)
print(running_time_ave_exp4)
###
###########################################################################################
###
gen_data = {"n":1000000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
FSA_1000000_mat = onlineFSA_numexp(gen_data, 0.01, 0, 100)
DR_exp5 = np.sum(FSA_1000000_mat[:,2] == gen_data["k"])
PCD_exp5 = np.mean(FSA_1000000_mat[:,2] / gen_data["k"])
rmse_ave_exp5 = np.mean(FSA_1000000_mat[:,3])
running_time_ave_exp5 = np.mean(FSA_1000000_mat[:,4])
print(DR_exp5)
print(PCD_exp5)
print(rmse_ave_exp5)
print(running_time_ave_exp5)
###
########################################################################################
### Big data experiment
########################################################################################
#######################
### beta = 1
#######################
gen_data = {"n":3000, "p":10000, "k":1000, "alpha":1, "beta_star":1, "dat_type":1}
FSA_3000_mat = onlineFSA_numexp(gen_data, 0.001, 0, 20)
DR_exp1 = np.sum(FSA_3000_mat[:,2] == gen_data["k"])
PCD_exp1 = np.mean(FSA_3000_mat[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(FSA_3000_mat[:,3])
running_time_ave_exp1 = np.mean(FSA_3000_mat[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
########################################################################################
###
gen_data = {"n":10000, "p":10000, "k":1000, "alpha":1, "beta_star":1, "dat_type":1}
FSA_10000_mat = onlineFSA_numexp(gen_data, 0.001, 0, 20)
DR_exp2 = np.sum(FSA_10000_mat[:,2] == gen_data["k"])
PCD_exp2 = np.mean(FSA_10000_mat[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(FSA_10000_mat[:,3])
running_time_ave_exp2 = np.mean(FSA_10000_mat[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
#########################################################################################
###
gen_data = {"n":30000, "p":10000, "k":1000, "alpha":1, "beta_star":1, "dat_type":1}
FSA_30000_mat = onlineFSA_numexp(gen_data, 0.001, 0, 20)
DR_exp3 = np.sum(FSA_30000_mat[:,2] == gen_data["k"])
PCD_exp3 = np.mean(FSA_30000_mat[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(FSA_30000_mat[:,3])
running_time_ave_exp3 = np.mean(FSA_30000_mat[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
####
#########################################################################################
####
gen_data = {"n":100000, "p":10000, "k":1000, "alpha":1, "beta_star":1, "dat_type":1}
FSA_100000_mat = onlineFSA_numexp(gen_data, 0.001, 0, 20)
DR_exp4 = np.sum(FSA_100000_mat[:,2] == gen_data["k"])
PCD_exp4 = np.mean(FSA_100000_mat[:,2] / gen_data["k"])
rmse_ave_exp4 = np.mean(FSA_100000_mat[:,3])
running_time_ave_exp4 = np.mean(FSA_100000_mat[:,4])
print(DR_exp4)
print(PCD_exp4)
print(rmse_ave_exp4)
print(running_time_ave_exp4)
#####
########################################################################################
#####
#################
### beta = 0.1
#################
gen_data = {"n":3000, "p":10000, "k":1000, "alpha":1, "beta_star":0.1, "dat_type":1}
FSA_beta01_3000_mat = onlineFSA_numexp(gen_data, 0.001, 0, 20)
DR_exp1 = np.sum(FSA_beta01_3000_mat[:,2] == gen_data["k"])
PCD_exp1 = np.mean(FSA_beta01_3000_mat[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(FSA_beta01_3000_mat[:,3])
running_time_ave_exp1 = np.mean(FSA_beta01_3000_mat[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
########################################################################################
###
gen_data = {"n":10000, "p":10000, "k":1000, "alpha":1, "beta_star":0.1, "dat_type":1}
FSA_beta01_10000_mat = onlineFSA_numexp(gen_data, 0.001, 0, 20)
DR_exp2 = np.sum(FSA_beta01_10000_mat[:,2] == gen_data["k"])
PCD_exp2 = np.mean(FSA_beta01_10000_mat[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(FSA_beta01_10000_mat[:,3])
running_time_ave_exp2 = np.mean(FSA_beta01_10000_mat[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
########################################################################################
###
gen_data = {"n":30000, "p":10000, "k":1000, "alpha":1, "beta_star":0.1, "dat_type":1}
FSA_beta01_30000_mat = onlineFSA_numexp(gen_data, 0.001, 0, 20)
DR_exp3 = np.sum(FSA_beta01_30000_mat[:,2] == gen_data["k"])
PCD_exp3 = np.mean(FSA_beta01_30000_mat[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(FSA_beta01_30000_mat[:,3])
running_time_ave_exp3 = np.mean(FSA_beta01_30000_mat[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
####
#########################################################################################
####
gen_data = {"n":100000, "p":10000, "k":1000, "alpha":1, "beta_star":0.1, "dat_type":1}
FSA_beta01_100000_mat = onlineFSA_numexp(gen_data, 0.001, 0, 20)
DR_exp4 = np.sum(FSA_beta01_100000_mat[:,2] == gen_data["k"])
PCD_exp4 = np.mean(FSA_beta01_100000_mat[:,2] / gen_data["k"])
rmse_ave_exp4 = np.mean(FSA_beta01_100000_mat[:,3])
running_time_ave_exp4 = np.mean(FSA_beta01_100000_mat[:,4])
print(DR_exp4)
print(PCD_exp4)
print(rmse_ave_exp4)
print(running_time_ave_exp4)
###
########################################################################################
###
################
###beta = 0.01
################
gen_data = {"n":10000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
FSA_beta001_10000_mat = onlineFSA_numexp(gen_data, 0.001, 0, 20)
DR_exp1 = np.sum(FSA_beta001_10000_mat[:,2] == gen_data["k"])
PCD_exp1 = np.mean(FSA_beta001_10000_mat[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(FSA_beta001_10000_mat[:,3])
running_time_ave_exp1 = np.mean(FSA_beta001_10000_mat[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
#########################################################################################
###
gen_data = {"n":30000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
FSA_beta001_30000_mat = onlineFSA_numexp(gen_data, 0.001, 0, 20)
DR_exp2 = np.sum(FSA_beta001_30000_mat[:,2] == gen_data["k"])
PCD_exp2 = np.mean(FSA_beta001_30000_mat[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(FSA_beta001_30000_mat[:,3])
running_time_ave_exp2 = np.mean(FSA_beta001_30000_mat[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
#############################################################################################
###
gen_data = {"n":100000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
FSA_beta001_100000_mat = onlineFSA_numexp(gen_data, 0.001, 0, 20)
DR_exp3 = np.sum(FSA_beta001_100000_mat[:,2] == gen_data["k"])
PCD_exp3 = np.mean(FSA_beta001_100000_mat[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(FSA_beta001_100000_mat[:,3])
running_time_ave_exp3 = np.mean(FSA_beta001_100000_mat[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
#########################################################################################
###
gen_data = {"n":300000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
FSA_beta001_300000_mat = onlineFSA_numexp(gen_data, 0.001, 0, 20)
DR_exp4 = np.sum(FSA_beta001_300000_mat[:,2] == gen_data["k"])
PCD_exp4 = np.mean(FSA_beta001_300000_mat[:,2] / gen_data["k"])
rmse_ave_exp4 = np.mean(FSA_beta001_300000_mat[:,3])
running_time_ave_exp4 = np.mean(FSA_beta001_300000_mat[:,4])
print(DR_exp4)
print(PCD_exp4)
print(rmse_ave_exp4)
print(running_time_ave_exp4)
####
#########################################################################################
####
gen_data = {"n":1000000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
FSA_beta001_1000000_mat = onlineFSA_numexp(gen_data, 0.001, 0, 20)
DR_exp5 = np.sum(FSA_beta001_1000000_mat[:,2] == gen_data["k"])
PCD_exp5 = np.mean(FSA_beta001_1000000_mat[:,2] / gen_data["k"])
rmse_ave_exp5 = np.mean(FSA_beta001_1000000_mat[:,3])
running_time_ave_exp5 = np.mean(FSA_beta001_1000000_mat[:,4])
print(DR_exp5)
print(PCD_exp5)
print(rmse_ave_exp5)
print(running_time_ave_exp5)