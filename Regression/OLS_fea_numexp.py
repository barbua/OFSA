# -*- coding: utf-8 -*-
#################################
## OLS feature selection ranking beta
## Lizhe Sun
#################################
#################################
### Load Package
#################################
import numpy as np
import time
import sys
sys.path.append("/Users/lizhesun/Documents/OFSelection_2023/simulations/regression")
import onlineFSA
import datageneration
#################################
#################################
#################################
#################################
### lbd is shrinkage
#################################
def OLS_fea_numexp(gen_dat, loop_time, lbd):
    ###
    ## Parameters
    ###
    n = gen_dat["n"]
    p = gen_dat["p"]
    k = gen_dat["k"]
    alpha = gen_dat["alpha"]
    beta_star = gen_dat["beta_star"]
    dat_type = gen_dat["dat_type"]
    result_mat = np.zeros((loop_time, 4))
    ##################
    ### Batch for data
    ##################
    if p > n:
        batch = n
    else:
        batch = p
    ###########
    num_batch = int(n / batch)
    ### Looping
    for i in range(loop_time):
        ###set seed
        np.random.seed(100 + i)
        ## Store seed
        result_mat[i, 0] = i + 100
        ########
        ### Generate data
        ### initial value for runningsums
        ########
        n_sum = 0
        Sx_sum = np.zeros((1, p))
        Sy_sum = 0
        Sxx_sum = np.zeros((p, p))
        Sxy_sum = np.zeros((p, 1))
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
        XX_normalize, XY_normalize, mu_x, mu_y, std_x = onlineFSA.standardize_ra(rs_sum)
        ##
        ##############################
        ##
        t_start = time.process_time()
        beta_hat_ols = onlineFSA.OLS_runningaves(XX_normalize, XY_normalize, lbd)
        beta_hat_ols = beta_hat_ols.ravel()
        beta_index = np.argsort(abs(beta_hat_ols))
        beta_index = beta_index.ravel()
        beta_index_de = beta_index[::-1]
        sel_index = beta_index_de[0:k]
        t_end = time.process_time()
        cost_time = t_end - t_start
        result_mat[i, 3] = cost_time
        num_true_var = len(np.intersect1d(istar, sel_index))
        result_mat[i, 1] = num_true_var
        ######
        ## test data
        ######
        ## RMSE
        ## Generate Test Data
        ######
        gen_testdata = {"n":batch, "p":p, "k":k, "alpha":alpha, "beta_star":beta_star, "dat_type":dat_type}
        testX, testY, betastar_vec, istar = datageneration.generate_data(gen_testdata)
        ################
        ### Standardize the test data
        ################
        testY_center = testY - mu_y * np.ones((batch, 1))
        testX_standardize = testX - np.ones((batch, 1)).dot(mu_x)
        inv_sigma = 1 / std_x
        testX_standardize = inv_sigma * testX_standardize
        #######
        ### refit by using OLS
        #######
        sel_index = sel_index.ravel()
        XX_sel = XX_normalize[np.ix_(sel_index, sel_index)]
        XY_sel = XY_normalize[sel_index]
        XX_sel_inv = np.linalg.inv(XX_sel + lbd * np.identity(k))
        beta_ols = XX_sel_inv.dot(XY_sel)
        beta_hat = np.zeros((p, 1))
        beta_hat[sel_index] = beta_ols
        ###
        ### rmse
        ###
        testY_hat = testX_standardize.dot(beta_hat)
        err_hat = testY_center.T - testY_hat.T
        rmse = np.sqrt(np.sum(err_hat**2) / batch) 
        result_mat[i, 2] = rmse
        #####
        #####
    return(result_mat)


########################################################################################
########################################################################################
####################
### Beta = 1
####################
gen_data = {"n":300, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
result_300_beta1 = OLS_fea_numexp(gen_data, 100, 0.05)
DR_exp0 = np.sum(result_300_beta1[:,1] == gen_data["k"])
PCD_exp0 = np.mean(result_300_beta1[:,1] / gen_data["k"])
rmse_ave_exp0 = np.mean(result_300_beta1[:,2])
running_time_ave_exp0 = np.mean(result_300_beta1[:,3])
print(DR_exp0)
print(PCD_exp0)
print(rmse_ave_exp0)
print(running_time_ave_exp0)
###
########################################################################################
###
gen_data = {"n":500, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
result_500_beta1 = OLS_fea_numexp(gen_data, 100, 0.05)
DR_exp1 = np.sum(result_500_beta1[:,1] == gen_data["k"])
PCD_exp1 = np.mean(result_500_beta1[:,1] / gen_data["k"])
rmse_ave_exp1 = np.mean(result_500_beta1[:,2])
running_time_ave_exp1 = np.mean(result_500_beta1[:,3])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
########################################################################################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
result_1000_beta1 = OLS_fea_numexp(gen_data, 100, 0.05)
DR_exp2 = np.sum(result_1000_beta1[:,1] == gen_data["k"])
PCD_exp2 = np.mean(result_1000_beta1[:,1] / gen_data["k"])
rmse_ave_exp2 = np.mean(result_1000_beta1[:,2])
running_time_ave_exp2 = np.mean(result_1000_beta1[:,3])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
########################################################################################
###
gen_data = {"n":3000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
result_3000_beta1 = OLS_fea_numexp(gen_data, 100, 0)
DR_exp3 = np.sum(result_3000_beta1[:,1] == gen_data["k"])
PCD_exp3 = np.mean(result_3000_beta1[:,1] / gen_data["k"])
rmse_ave_exp3 = np.mean(result_3000_beta1[:,2])
running_time_ave_exp3 = np.mean(result_3000_beta1[:,3])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
########################################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
result_10000_beta1 = OLS_fea_numexp(gen_data, 100, 0)
DR_exp4 = np.sum(result_10000_beta1[:,1] == gen_data["k"])
PCD_exp4 = np.mean(result_10000_beta1[:,1] / gen_data["k"])
rmse_ave_exp4 = np.mean(result_10000_beta1[:,2])
running_time_ave_exp4 = np.mean(result_10000_beta1[:,3])
print(DR_exp4)
print(PCD_exp4)
print(rmse_ave_exp4)
print(running_time_ave_exp4)
###
########################################################################################
###########################
### beta =0.1
###########################
########################################################################################
#gen_data = {"n":300, "p":1000, "k":100, "alpha":1, "beta_star":0.1, "dat_type":1}
#result_300_beta01 = OLS_fea_numexp(gen_data, 100, 0.01)
#DR_exp1 = np.sum(result_300_beta01[:,1] == gen_data["k"])
#PCD_exp1 = np.mean(result_300_beta01[:,1] / gen_data["k"])
#rmse_ave_exp1 = np.mean(result_300_beta01[:,2])
#running_time_ave_exp1 = np.mean(result_300_beta01[:,3])
#print(DR_exp1)
#print(PCD_exp1)
#print(rmse_ave_exp1)
#print(running_time_ave_exp1)
###
########################################################################################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":0.1, "dat_type":1}
result_1000_beta01 = OLS_fea_numexp(gen_data, 100, 0.01)
DR_exp2 = np.sum(result_1000_beta01[:,1] == gen_data["k"])
PCD_exp2 = np.mean(result_1000_beta01[:,1] / gen_data["k"])
rmse_ave_exp2 = np.mean(result_1000_beta01[:,2])
running_time_ave_exp2 = np.mean(result_1000_beta01[:,3])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
########################################################################################
###
gen_data = {"n":3000, "p":1000, "k":100, "alpha":1, "beta_star":0.1, "dat_type":1}
result_3000_beta01 = OLS_fea_numexp(gen_data, 100, 0)
DR_exp3 = np.sum(result_3000_beta01[:,1] == gen_data["k"])
PCD_exp3 = np.mean(result_3000_beta01[:,1] / gen_data["k"])
rmse_ave_exp3 = np.mean(result_3000_beta01[:,2])
running_time_ave_exp3 = np.mean(result_3000_beta01[:,3])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
########################################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":0.1, "dat_type":1}
result_10000_beta01 = OLS_fea_numexp(gen_data, 100, 0)
DR_exp4 = np.sum(result_10000_beta01[:,1] == gen_data["k"])
PCD_exp4 = np.mean(result_10000_beta01[:,1] / gen_data["k"])
rmse_ave_exp4 = np.mean(result_10000_beta01[:,2])
running_time_ave_exp4 = np.mean(result_10000_beta01[:,3])
print(DR_exp4)
print(PCD_exp4)
print(rmse_ave_exp4)
print(running_time_ave_exp4)
###
########################################################################################
###
##########################
## beta = 0.01
##########################
########################################################################################
gen_data = {"n":500, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
result_500_beta001 = OLS_fea_numexp(gen_data, 100, 0.05)
DR_exp0 = np.sum(result_500_beta001[:,1] == gen_data["k"])
PCD_exp0 = np.mean(result_500_beta001[:,1] / gen_data["k"])
rmse_ave_exp0 = np.mean(result_500_beta001[:,2])
running_time_ave_exp0 = np.mean(result_500_beta001[:,3])
print(DR_exp0)
print(PCD_exp0)
print(rmse_ave_exp0)
print(running_time_ave_exp0)
###
########################################################################################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
result_1000_beta001 = OLS_fea_numexp(gen_data, 100, 0.05)
DR_exp1 = np.sum(result_1000_beta001[:,1] == gen_data["k"])
PCD_exp1 = np.mean(result_1000_beta001[:,1] / gen_data["k"])
rmse_ave_exp1 = np.mean(result_1000_beta001[:,2])
running_time_ave_exp1 = np.mean(result_1000_beta001[:,3])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
########################################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
result_10000_beta001 = OLS_fea_numexp(gen_data, 100, 0)
DR_exp2 = np.sum(result_10000_beta001[:,1] == gen_data["k"])
PCD_exp2 = np.mean(result_10000_beta001[:,1] / gen_data["k"])
rmse_ave_exp2 = np.mean(result_10000_beta001[:,2])
running_time_ave_exp2 = np.mean(result_10000_beta001[:,3])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
########################################################################################
###
gen_data = {"n":100000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
result_100000_beta001 = OLS_fea_numexp(gen_data, 100, 0)
DR_exp3 = np.sum(result_100000_beta001[:,1] == gen_data["k"])
PCD_exp3 = np.mean(result_100000_beta001[:,1] / gen_data["k"])
rmse_ave_exp3 = np.mean(result_100000_beta001[:,2])
running_time_ave_exp3 = np.mean(result_100000_beta001[:,3])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
########################################################################################
###
gen_data = {"n":300000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
result_300000_beta001 = OLS_fea_numexp(gen_data, 100, 0)
DR_exp4 = np.sum(result_300000_beta001[:,1] == gen_data["k"])
PCD_exp4 = np.mean(result_300000_beta001[:,1] / gen_data["k"])
rmse_ave_exp4 = np.mean(result_300000_beta001[:,2])
running_time_ave_exp4 = np.mean(result_300000_beta001[:,3])
print(DR_exp4)
print(PCD_exp4)
print(rmse_ave_exp4)
print(running_time_ave_exp4)
###
########################################################################################
###
gen_data = {"n":1000000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
result_1000000_beta001 = OLS_fea_numexp(gen_data, 100, 0)
DR_exp5 = np.sum(result_1000000_beta001[:,1] == gen_data["k"])
PCD_exp5 = np.mean(result_1000000_beta001[:,1] / gen_data["k"])
rmse_ave_exp5 = np.mean(result_1000000_beta001[:,2])
running_time_ave_exp5 = np.mean(result_1000000_beta001[:,3])
print(DR_exp5)
print(PCD_exp5)
print(rmse_ave_exp5)
print(running_time_ave_exp5)
###
########################################################################################
###
###
########################################################################################
###
###########################
### Big data Experiment
###########################
###########################
### beta = 1
###########################
#gen_data = {"n":3000, "p":10000, "k":1000, "alpha":1, "beta_star":1, "dat_type":1}
#result_3000_beta1 = OLS_fea_numexp(gen_data, 20, 0.01)
#DR_exp1 = np.sum(result_3000_beta1[:,1] == gen_data["k"])
#PCD_exp1 = np.mean(result_3000_beta1[:,1] / gen_data["k"])
#rmse_ave_exp1 = np.mean(result_3000_beta1[:,2])
#running_time_ave_exp1 = np.mean(result_3000_beta1[:,3])
#print(DR_exp1)
#print(PCD_exp1)
#print(rmse_ave_exp1)
#print(running_time_ave_exp1)
###
########################################################################################
###
gen_data = {"n":10000, "p":10000, "k":1000, "alpha":1, "beta_star":1, "dat_type":1}
result_10000_beta1 = OLS_fea_numexp(gen_data, 20, 0.01)
DR_exp2 = np.sum(result_10000_beta1[:,1] == gen_data["k"])
PCD_exp2 = np.mean(result_10000_beta1[:,1] / gen_data["k"])
rmse_ave_exp2 = np.mean(result_10000_beta1[:,2])
running_time_ave_exp2 = np.mean(result_10000_beta1[:,3])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
########################################################################################
###
gen_data = {"n":30000, "p":10000, "k":1000, "alpha":1, "beta_star":1, "dat_type":1}
result_30000_beta1 = OLS_fea_numexp(gen_data, 20, 0)
DR_exp3 = np.sum(result_30000_beta1[:,1] == gen_data["k"])
PCD_exp3 = np.mean(result_30000_beta1[:,1] / gen_data["k"])
rmse_ave_exp3 = np.mean(result_30000_beta1[:,2])
running_time_ave_exp3 = np.mean(result_30000_beta1[:,3])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
########################################################################################
###
gen_data = {"n":100000, "p":10000, "k":1000, "alpha":1, "beta_star":1, "dat_type":1}
result_100000_beta1 = OLS_fea_numexp(gen_data, 20, 0)
DR_exp4 = np.sum(result_100000_beta1[:,1] == gen_data["k"])
PCD_exp4 = np.mean(result_100000_beta1[:,1] / gen_data["k"])
rmse_ave_exp4 = np.mean(result_100000_beta1[:,2])
running_time_ave_exp4 = np.mean(result_100000_beta1[:,3])
print(DR_exp4)
print(PCD_exp4)
print(rmse_ave_exp4)
print(running_time_ave_exp4)
###
########################################################################################
###
######################
### beta = 0.1
######################
###
#gen_data = {"n":3000, "p":10000, "k":1000, "alpha":1, "beta_star":0.1, "dat_type":1}
#result_3000_beta01 = OLS_fea_numexp(gen_data, 20, 0.01)
#DR_exp1 = np.sum(result_3000_beta01[:,1] == gen_data["k"])
#PCD_exp1 = np.mean(result_3000_beta01[:,1] / gen_data["k"])
#rmse_ave_exp1 = np.mean(result_3000_beta01[:,2])
#running_time_ave_exp1 = np.mean(result_3000_beta01[:,3])
#print(DR_exp1)
#print(PCD_exp1)
#print(rmse_ave_exp1)
#print(running_time_ave_exp1)
###
########################################################################################
###
gen_data = {"n":10000, "p":10000, "k":1000, "alpha":1, "beta_star":0.1, "dat_type":1}
result_10000_beta01 = OLS_fea_numexp(gen_data, 20, 0.01)
DR_exp2 = np.sum(result_10000_beta01[:,1] == gen_data["k"])
PCD_exp2 = np.mean(result_10000_beta01[:,1] / gen_data["k"])
rmse_ave_exp2 = np.mean(result_10000_beta01[:,2])
running_time_ave_exp2 = np.mean(result_10000_beta01[:,3])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
########################################################################################
###
gen_data = {"n":30000, "p":10000, "k":1000, "alpha":1, "beta_star":0.1, "dat_type":1}
result_30000_beta01 = OLS_fea_numexp(gen_data, 20, 0)
DR_exp3 = np.sum(result_30000_beta01[:,1] == gen_data["k"])
PCD_exp3 = np.mean(result_30000_beta01[:,1] / gen_data["k"])
rmse_ave_exp3 = np.mean(result_30000_beta01[:,2])
running_time_ave_exp3 = np.mean(result_30000_beta01[:,3])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
########################################################################################
###
gen_data = {"n":100000, "p":10000, "k":1000, "alpha":1, "beta_star":0.1, "dat_type":1}
result_100000_beta01 = OLS_fea_numexp(gen_data, 20, 0)
DR_exp4 = np.sum(result_100000_beta01[:,1] == gen_data["k"])
PCD_exp4 = np.mean(result_100000_beta01[:,1] / gen_data["k"])
rmse_ave_exp4 = np.mean(result_100000_beta01[:,2])
running_time_ave_exp4 = np.mean(result_100000_beta01[:,3])
print(DR_exp4)
print(PCD_exp4)
print(rmse_ave_exp4)
print(running_time_ave_exp4)
###
########################################################################################
###
######################
### beta = 0.01
######################
###
gen_data = {"n":10000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
result_10000_beta001 = OLS_fea_numexp(gen_data, 20, 0.01)
DR_exp1 = np.sum(result_10000_beta001[:,1] == gen_data["k"])
PCD_exp1 = np.mean(result_10000_beta001[:,1] / gen_data["k"])
rmse_ave_exp1 = np.mean(result_10000_beta001[:,2])
running_time_ave_exp1 = np.mean(result_10000_beta001[:,3])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
########################################################################################
###
gen_data = {"n":30000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
result_30000_beta001 = OLS_fea_numexp(gen_data, 20, 0)
DR_exp2 = np.sum(result_30000_beta001[:,1] == gen_data["k"])
PCD_exp2 = np.mean(result_30000_beta001[:,1] / gen_data["k"])
rmse_ave_exp2 = np.mean(result_30000_beta001[:,2])
running_time_ave_exp2 = np.mean(result_30000_beta001[:,3])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
########################################################################################
###
gen_data = {"n":100000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
result_100000_beta001 = OLS_fea_numexp(gen_data, 20, 0)
DR_exp3 = np.sum(result_100000_beta001[:,1] == gen_data["k"])
PCD_exp3 = np.mean(result_100000_beta001[:,1] / gen_data["k"])
rmse_ave_exp3 = np.mean(result_100000_beta001[:,2])
running_time_ave_exp3 = np.mean(result_100000_beta001[:,3])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
########################################################################################
###
gen_data = {"n":300000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
result_300000_beta001 = OLS_fea_numexp(gen_data, 20, 0)
DR_exp4 = np.sum(result_300000_beta001[:,1] == gen_data["k"])
PCD_exp4 = np.mean(result_300000_beta001[:,1] / gen_data["k"])
rmse_ave_exp4 = np.mean(result_300000_beta001[:,2])
running_time_ave_exp4 = np.mean(result_300000_beta001[:,3])
print(DR_exp4)
print(PCD_exp4)
print(rmse_ave_exp4)
print(running_time_ave_exp4)
###
########################################################################################
###
gen_data = {"n":1000000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
result_1000000_beta001 = OLS_fea_numexp(gen_data, 20, 0)
DR_exp5 = np.sum(result_1000000_beta001[:,1] == gen_data["k"])
PCD_exp5 = np.mean(result_1000000_beta001[:,1] / gen_data["k"])
rmse_ave_exp5 = np.mean(result_1000000_beta001[:,2])
running_time_ave_exp5 = np.mean(result_1000000_beta001[:,3])
print(DR_exp5)
print(PCD_exp5)
print(rmse_ave_exp5)
print(running_time_ave_exp5)
###
########################################################################################
###