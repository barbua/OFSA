# -*- coding: utf-8 -*-
##########################################
## SIHT feature selection
## Lizhe Sun
##########################################
####################
## Load package
####################
import time
import sys
sys.path.append("/Users/lizhesun/Documents/OFSelection_2023/simulations/regression")
import SGD
import datageneration
import numpy as np
########################################################################################
########################################################################################
########################################################################################
########################################################################################
def SIHT_num_exp(gen_data, eta, minibatch, loop_times):
    ################
    n = gen_data["n"]
    p = gen_data["p"]
    k = gen_data["k"]
    alpha = gen_data["alpha"]
    beta_star = gen_data["beta_star"]
    dat_type = gen_data["dat_type"]
    result_mat = np.zeros((loop_times, 5))
    ### Looping
    for i in range(loop_times):
        ### Generate seed
        np.random.seed(i + 100)
        result_mat[i,0] = i + 100
        result_mat[i, 1] = eta
        ###
        batch = p
        outer_loop = int(n / batch)
        beta = np.zeros((p, 1))
        beta0 = 0
        time_total = 0
        for j in range(outer_loop):
            ############
            ### generate data
            ############
            gen_data_bat = {"n":batch, "p":p, "k":k, "alpha":alpha, "beta_star":beta_star, "dat_type":dat_type}
            Xtr_batch, Ytr_batch, betastar_vec, istar = datageneration.generate_data(gen_data_bat)
            #############################
            eta_ada = eta / np.sqrt(j+1)
            #############################
            ### SIHT algorithm
            #############################
            t_start = time.process_time()
            beta, beta0, sel_index = SGD.train_SIHT(Xtr_batch, Ytr_batch, beta, beta0, eta_ada, k, minibatch)
            t_end = time.process_time()
            t_cost = t_end - t_start
            time_total = time_total + t_cost
            
        result_mat[i, 4] = time_total
        ###########
        ## Compare the index with the true index
        ###########
        num_true_var = len(np.intersect1d(istar, sel_index))
        result_mat[i, 2] = num_true_var
        #########
        ### Generate test data
        #########
        gen_testdata = {"n":batch, "p":p, "k":k, "alpha":alpha, "beta_star":beta_star, "dat_type":dat_type}
        testX, testY, betastar_vec, istar = datageneration.generate_data(gen_testdata)
        ########
        ### RMSE
        ########
        testY_hat = testX.dot(beta) + beta0 * np.ones((batch, 1))
        err_hat = testY.T - testY_hat.T
        rmse = np.sqrt(np.sum(err_hat**2) / batch) 
        result_mat[i, 3] = rmse
        ########
    return(result_mat)
########################################################################################
####################### Simulation #####################################################
########################################################################################
###
##############################################################################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
SIHT_1000_beta1 = SIHT_num_exp(gen_data, 0.0005, 25, 100)
DR_exp1 = np.sum(SIHT_1000_beta1[:,2] == gen_data["k"])
PCD_exp1 = np.mean(SIHT_1000_beta1[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(SIHT_1000_beta1[:,3])
running_time_ave_exp1 = np.mean(SIHT_1000_beta1[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
###############################################################################
###
gen_data = {"n":3000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
SIHT_3000_beta1 = SIHT_num_exp(gen_data, 0.0005, 25, 100)
DR_exp2 = np.sum(SIHT_3000_beta1[:,2] == gen_data["k"])
PCD_exp2 = np.mean(SIHT_3000_beta1[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(SIHT_3000_beta1[:,3])
running_time_ave_exp2 = np.mean(SIHT_3000_beta1[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
########################################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
SIHT_10000_beta1 = SIHT_num_exp(gen_data, 0.0005, 25, 100)
DR_exp3 = np.sum(SIHT_10000_beta1[:,2] == gen_data["k"])
PCD_exp3 = np.mean(SIHT_10000_beta1[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(SIHT_10000_beta1[:,3])
running_time_ave_exp3 = np.mean(SIHT_10000_beta1[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
########################################################################################
###
#####################
### beta = 0.1
#####################
###
##############################################################################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":0.1, "dat_type":1}
SIHT_1000_beta01 = SIHT_num_exp(gen_data, 0.0005, 25, 100)
DR_exp1 = np.sum(SIHT_1000_beta01[:,2] == gen_data["k"])
PCD_exp1 = np.mean(SIHT_1000_beta01[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(SIHT_1000_beta01[:,3])
running_time_ave_exp1 = np.mean(SIHT_1000_beta01[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
###############################################################################
###
gen_data = {"n":3000, "p":1000, "k":100, "alpha":1, "beta_star":0.1, "dat_type":1}
SIHT_3000_beta01 = SIHT_num_exp(gen_data, 0.0005, 25, 100)
DR_exp2 = np.sum(SIHT_3000_beta01[:,2] == gen_data["k"])
PCD_exp2 = np.mean(SIHT_3000_beta01[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(SIHT_3000_beta01[:,3])
running_time_ave_exp2 = np.mean(SIHT_3000_beta01[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
#########################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":0.1, "dat_type":1}
SIHT_10000_beta01 = SIHT_num_exp(gen_data, 0.0005, 25, 100)
DR_exp3 = np.sum(SIHT_10000_beta01[:,2] == gen_data["k"])
PCD_exp3 = np.mean(SIHT_10000_beta01[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(SIHT_10000_beta01[:,3])
running_time_ave_exp3 = np.mean(SIHT_10000_beta01[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
########################################################################################
###
#####################
### beta = 0.01
#####################
###
########################################################################################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
SIHT_1000_beta001 = SIHT_num_exp(gen_data, 0.0005, 25, 100)
DR_exp1 = np.sum(SIHT_1000_beta001[:,2] == gen_data["k"])
PCD_exp1 = np.mean(SIHT_1000_beta001[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(SIHT_1000_beta001[:,3])
running_time_ave_exp1 = np.mean(SIHT_1000_beta001[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
########################################################################################
####
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
SIHT_10000_beta001 = SIHT_num_exp(gen_data, 0.0005, 25, 100)
DR_exp2 = np.sum(SIHT_10000_beta001[:,2] == gen_data["k"])
PCD_exp2 = np.mean(SIHT_10000_beta001[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(SIHT_10000_beta001[:,3])
running_time_ave_exp2 = np.mean(SIHT_10000_beta001[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
########################################################################################
###
gen_data = {"n":100000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
SIHT_100000_beta001 = SIHT_num_exp(gen_data, 0.0005, 25, 100)
DR_exp3 = np.sum(SIHT_100000_beta001[:,2] == gen_data["k"])
PCD_exp3 = np.mean(SIHT_100000_beta001[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(SIHT_100000_beta001[:,3])
running_time_ave_exp3 = np.mean(SIHT_100000_beta001[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
########################################################################################
###
gen_data = {"n":300000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
SIHT_300000_beta001 = SIHT_num_exp(gen_data, 0.0005, 25, 100)
DR_exp4 = np.sum(SIHT_300000_beta001[:,2] == gen_data["k"])
PCD_exp4 = np.mean(SIHT_300000_beta001[:,2] / gen_data["k"])
rmse_ave_exp4 = np.mean(SIHT_300000_beta001[:,3])
running_time_ave_exp4 = np.mean(SIHT_300000_beta001[:,4])
print(DR_exp4)
print(PCD_exp4)
print(rmse_ave_exp4)
print(running_time_ave_exp4)
###
########################################################################################
###
gen_data = {"n":1000000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
SIHT_1000000_beta001 = SIHT_num_exp(gen_data, 0.0005, 25, 100)
DR_exp5 = np.sum(SIHT_1000000_beta001[:,2] == gen_data["k"])
PCD_exp5 = np.mean(SIHT_1000000_beta001[:,2] / gen_data["k"])
rmse_ave_exp5 = np.mean(SIHT_1000000_beta001[:,3])
running_time_ave_exp5 = np.mean(SIHT_1000000_beta001[:,4])
print(DR_exp5)
print(PCD_exp5)
print(rmse_ave_exp5)
print(running_time_ave_exp5)
###
########################################################################################
###
############################
### Big data experiment
############################
###############
## beta = 1
###############
###
########################################################################################
###
gen_data = {"n":10000, "p":10000, "k":1000, "alpha":1, "beta_star":1, "dat_type":1}
SIHT_10000_beta1 = SIHT_num_exp(gen_data, 0.0005, 25, 20)
DR_exp1 = np.sum(SIHT_10000_beta1[:,2] == gen_data["k"])
PCD_exp1 = np.mean(SIHT_10000_beta1[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(SIHT_10000_beta1[:,3])
running_time_ave_exp1 = np.mean(SIHT_10000_beta1[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
########################################################################################
###
gen_data = {"n":30000, "p":10000, "k":1000, "alpha":1, "beta_star":1, "dat_type":1}
SIHT_30000_beta1 = SIHT_num_exp(gen_data, 0.0005, 25, 20)
DR_exp2 = np.sum(SIHT_30000_beta1[:,2] == gen_data["k"])
PCD_exp2 = np.mean(SIHT_30000_beta1[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(SIHT_30000_beta1[:,3])
running_time_ave_exp2 = np.mean(SIHT_30000_beta1[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
########################################################################################
###
gen_data = {"n":100000, "p":10000, "k":1000, "alpha":1, "beta_star":1, "dat_type":1}
SIHT_100000_beta1 = SIHT_num_exp(gen_data, 0.0005, 25, 20)
DR_exp3 = np.sum(SIHT_100000_beta1[:,2] == gen_data["k"])
PCD_exp3 = np.mean(SIHT_100000_beta1[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(SIHT_100000_beta1[:,3])
running_time_ave_exp3 = np.mean(SIHT_100000_beta1[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
########################################################################################
###
###############
## beta = 0.1
###############
###
########################################################################################
###
gen_data = {"n":10000, "p":10000, "k":1000, "alpha":1, "beta_star":0.1, "dat_type":1}
SIHT_10000_beta01 = SIHT_num_exp(gen_data, 0.0005, 25, 20)
DR_exp1 = np.sum(SIHT_10000_beta01[:,2] == gen_data["k"])
PCD_exp1 = np.mean(SIHT_10000_beta01[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(SIHT_10000_beta01[:,3])
running_time_ave_exp1 = np.mean(SIHT_10000_beta01[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
########################################################################################
###
gen_data = {"n":30000, "p":10000, "k":1000, "alpha":1, "beta_star":0.1, "dat_type":1}
SIHT_30000_beta01 = SIHT_num_exp(gen_data, 0.0005, 25, 20)
DR_exp2 = np.sum(SIHT_30000_beta01[:,2] == gen_data["k"])
PCD_exp2 = np.mean(SIHT_30000_beta01[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(SIHT_30000_beta01[:,3])
running_time_ave_exp2 = np.mean(SIHT_30000_beta01[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
########################################################################################
###
gen_data = {"n":100000, "p":10000, "k":1000, "alpha":1, "beta_star":0.1, "dat_type":1}
SIHT_100000_beta01 = SIHT_num_exp(gen_data, 0.0005, 25, 20)
DR_exp3 = np.sum(SIHT_100000_beta01[:,2] == gen_data["k"])
PCD_exp3 = np.mean(SIHT_100000_beta01[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(SIHT_100000_beta01[:,3])
running_time_ave_exp3 = np.mean(SIHT_100000_beta01[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
########################################################################################
###
#####################
### beta = 0.01
#####################
###
########################################################################################
###
gen_data = {"n":10000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
SIHT_10000_beta001 = SIHT_num_exp(gen_data, 0.0005, 25, 20)
DR_exp1 = np.sum(SIHT_10000_beta001[:,2] == gen_data["k"])
PCD_exp1 = np.mean(SIHT_10000_beta001[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(SIHT_10000_beta001[:,3])
running_time_ave_exp1 = np.mean(SIHT_10000_beta001[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
########################################################################################
###
gen_data = {"n":30000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
SIHT_30000_beta001 = SIHT_num_exp(gen_data, 0.0005, 25, 20)
DR_exp2 = np.sum(SIHT_30000_beta001[:,2] == gen_data["k"])
PCD_exp2 = np.mean(SIHT_30000_beta001[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(SIHT_30000_beta001[:,3])
running_time_ave_exp2 = np.mean(SIHT_30000_beta001[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
########################################################################################
###
gen_data = {"n":100000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
SIHT_100000_beta001 = SIHT_num_exp(gen_data, 0.0005, 25, 20)
DR_exp3 = np.sum(SIHT_100000_beta001[:,2] == gen_data["k"])
PCD_exp3 = np.mean(SIHT_100000_beta001[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(SIHT_100000_beta001[:,3])
running_time_ave_exp3 = np.mean(SIHT_100000_beta001[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
########################################################################################
###
gen_data = {"n":300000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
SIHT_300000_beta001 = SIHT_num_exp(gen_data, 0.0005, 25, 20)
DR_exp4 = np.sum(SIHT_300000_beta001[:,2] == gen_data["k"])
PCD_exp4 = np.mean(SIHT_300000_beta001[:,2] / gen_data["k"])
rmse_ave_exp4 = np.mean(SIHT_300000_beta001[:,3])
running_time_ave_exp4 = np.mean(SIHT_300000_beta001[:,4])
print(DR_exp4)
print(PCD_exp4)
print(rmse_ave_exp4)
print(running_time_ave_exp4)
###
########################################################################################
###
gen_data = {"n":1000000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
SIHT_1000000_beta001 = SIHT_num_exp(gen_data, 0.0005, 25, 20)
DR_exp5 = np.sum(SIHT_1000000_beta001[:,2] == gen_data["k"])
PCD_exp5 = np.mean(SIHT_1000000_beta001[:,2] / gen_data["k"])
rmse_ave_exp5 = np.mean(SIHT_1000000_beta001[:,3])
running_time_ave_exp5 = np.mean(SIHT_1000000_beta001[:,4])
print(DR_exp5)
print(PCD_exp5)
print(rmse_ave_exp5)
print(running_time_ave_exp5)
###