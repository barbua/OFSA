# -*- coding: utf-8 -*-
##########################################
## SGD feature selection
## Lizhe Sun
##########################################
####################
## Load package
####################
import SGD
import datageneration
import numpy as np
import pandas as pd
import time
########################################################################################
########################################################################################
########################################################################################
def SGD_num_exp(gen_data, eta, mb_size, exp_times):
    #######################
    n = gen_data["n"]
    p = gen_data["p"]
    k = gen_data["k"]
    alpha = gen_data["alpha"]
    beta_star = gen_data["beta_star"]
    dat_type = gen_data["dat_type"]
    ########################
    ### Set initial value
    ########################
    result_mat = np.zeros((exp_times, 5))
    batch = p
    outer_loop = int(n / batch)
    for i in range(exp_times):
        ### Generate seed
        np.random.seed(i + 100)
        result_mat[i,0] = i + 100
        result_mat[i, 1] = eta
        total_cost = 0
        beta = np.zeros((p, 1))
        beta0 = 0
        ################################################################################
        for j in range(outer_loop):
            ### Generate one batch data
            gen_data_bat = {"n":batch, "p":p, "k":k, "alpha":alpha, "beta_star":beta_star, "dat_type":dat_type}
            Xtr_batch, Ytr_batch, betastar_vec, istar = datageneration.generate_data(gen_data_bat)
            #############################
            eta_ada = eta / np.sqrt(j + 1)
            #############################
            t_start = time.time()
            beta, beta0 = SGD.train_SGD(Xtr_batch, Ytr_batch, beta, beta0, eta_ada, mb_size)
            t_end = time.time()
            t_cost = t_end - t_start
            total_cost = total_cost + t_cost
        ###########
        ## Compare the index with the true index
        ###########
        result_mat[i, 4] = total_cost
        result_mat[i, 2] = 0
        #########
        ### Generate test data
        #########
        gen_testdata = {"n":1000, "p":p, "k":k, "alpha":alpha, "beta_star":beta_star, "dat_type":dat_type}
        testX, testY, betastar_vec, istar = datageneration.generate_data(gen_testdata)
        ########
        ### RMSE
        ########
        testY_hat = testX.dot(beta) + beta0 * np.ones((1000, 1))
        err_hat = testY.T - testY_hat.T
        rmse = np.sqrt(np.sum(err_hat**2) / 1000) 
        result_mat[i, 3] = rmse
        #print(i)
        ######
    return(result_mat)
    
    
    
########################################################################################
####################### Simulation #####################################################
########################################################################################
###
##############################################################################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
SGD_1000_mat = SGD_num_exp(gen_data, 0.0005, 25, 100)
SGD_1000_dat = pd.DataFrame(SGD_1000_mat)
SGD_1000_dat.to_csv('/Users/lizhesun/Documents/Project_One_IncreasementRegression/SGD_result/beta1/SGD_1000_1000_dat.txt', sep = ',')
DR_exp1 = np.sum(SGD_1000_mat[:,2] == gen_data["k"])
PCD_exp1 = np.mean(SGD_1000_mat[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(SGD_1000_mat[:,3])
running_time_ave_exp1 = np.mean(SGD_1000_mat[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
###############################################################################
###
gen_data = {"n":3000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
SGD_3000_mat = SGD_num_exp(gen_data, 0.0005, 25, 100)
SGD_3000_dat = pd.DataFrame(SGD_3000_mat)
SGD_3000_dat.to_csv('/Users/lizhesun/Documents/Project_One_IncreasementRegression/SGD_result/beta1/SGD_3000_1000_dat.txt', sep = ',')
DR_exp2 = np.sum(SGD_3000_mat[:,2] == gen_data["k"])
PCD_exp2 = np.mean(SGD_3000_mat[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(SGD_3000_mat[:,3])
running_time_ave_exp2 = np.mean(SGD_3000_mat[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
#########################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
SGD_10000_mat = SGD_num_exp(gen_data, 0.0005, 25, 100)
SGD_10000_dat = pd.DataFrame(SGD_10000_mat)
SGD_10000_dat.to_csv('/Users/lizhesun/Documents/Project_One_IncreasementRegression/SGD_result/beta1/SGD_10000_1000_dat.txt', sep = ',')
DR_exp3 = np.sum(SGD_10000_mat[:,2] == gen_data["k"])
PCD_exp3 = np.mean(SGD_10000_mat[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(SGD_10000_mat[:,3])
running_time_ave_exp3 = np.mean(SGD_10000_mat[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
########################################################################
###
#####################
### beta = 0.1
#####################
###
##############################################################################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":0.1, "dat_type":1}
SGD_1000_mat = SGD_num_exp(gen_data, 0.0005, 25, 100)
SGD_1000_dat = pd.DataFrame(SGD_1000_mat)
SGD_1000_dat.to_csv('/Users/lizhesun/Documents/Project_One_IncreasementRegression/SGD_result/beta1/SGD_1000_1000_beta01_dat.txt', sep = ',')
DR_exp1 = np.sum(SGD_1000_mat[:,2] == gen_data["k"])
PCD_exp1 = np.mean(SGD_1000_mat[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(SGD_1000_mat[:,3])
running_time_ave_exp1 = np.mean(SGD_1000_mat[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
###############################################################################
###
gen_data = {"n":3000, "p":1000, "k":100, "alpha":1, "beta_star":0.1, "dat_type":1}
SGD_3000_mat = SGD_num_exp(gen_data, 0.0005, 25, 100)
SGD_3000_dat = pd.DataFrame(SGD_3000_mat)
SGD_3000_dat.to_csv('/Users/lizhesun/Documents/Project_One_IncreasementRegression/SGD_result/beta1/SGD_3000_1000_beta01_dat.txt', sep = ',')
DR_exp2 = np.sum(SGD_3000_mat[:,2] == gen_data["k"])
PCD_exp2 = np.mean(SGD_3000_mat[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(SGD_3000_mat[:,3])
running_time_ave_exp2 = np.mean(SGD_3000_mat[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
#########################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":0.1, "dat_type":1}
SGD_10000_mat = SGD_num_exp(gen_data, 0.0005, 25, 100)
SGD_10000_dat = pd.DataFrame(SGD_10000_mat)
SGD_10000_dat.to_csv('/Users/lizhesun/Documents/Project_One_IncreasementRegression/SGD_result/beta1/SGD_10000_1000_beta01_dat.txt', sep = ',')
DR_exp3 = np.sum(SGD_10000_mat[:,2] == gen_data["k"])
PCD_exp3 = np.mean(SGD_10000_mat[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(SGD_10000_mat[:,3])
running_time_ave_exp3 = np.mean(SGD_10000_mat[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
###########################################################################
###
#####################
### beta = 0.01
#####################
###
########################################################################################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
SGD_1000_mat = SGD_num_exp(gen_data, 0.0005, 25, 100)
SGD_1000_dat = pd.DataFrame(SGD_1000_mat)
SGD_1000_dat.to_csv('/Users/lizhesun/Documents/Project_One_IncreasementRegression/SGD_result/beta1/SGD_1000_1000_beta001_dat.txt', sep = ',')
DR_exp1 = np.sum(SGD_1000_mat[:,2] == gen_data["k"])
PCD_exp1 = np.mean(SGD_1000_mat[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(SGD_1000_mat[:,3])
running_time_ave_exp1 = np.mean(SGD_1000_mat[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
########################################################################################
####
#gen_data = {"n":3000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
#SGD_3000_mat = SGD_num_exp(gen_data, 0.0005, 25, 100)
#SGD_3000_dat = pd.DataFrame(SGD_3000_mat)
#SGD_3000_dat.to_csv('/Users/lizhesun/Documents/Project_One_IncreasementRegression/SGD_result/beta1/SGD_3000_1000_beta001_dat.txt', sep = ',')
#DR_exp2 = np.sum(SGD_3000_mat[:,2] == gen_data["k"])
#PCD_exp2 = np.mean(SGD_3000_mat[:,2] / gen_data["k"])
#rmse_ave_exp2 = np.mean(SGD_3000_mat[:,3])
#running_time_ave_exp2 = np.mean(SGD_3000_mat[:,4])
#print(DR_exp2)
#print(PCD_exp2)
#print(rmse_ave_exp2)
#print(running_time_ave_exp2)
###
########################################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
SGD_10000_mat = SGD_num_exp(gen_data, 0.0005, 25, 100)
SGD_10000_dat = pd.DataFrame(SGD_10000_mat)
SGD_10000_dat.to_csv('/Users/lizhesun/Documents/Project_One_IncreasementRegression/SGD_result/beta1/SGD_10000_1000_beta001_dat.txt', sep = ',')
DR_exp3 = np.sum(SGD_10000_mat[:,2] == gen_data["k"])
PCD_exp3 = np.mean(SGD_10000_mat[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(SGD_10000_mat[:,3])
running_time_ave_exp3 = np.mean(SGD_10000_mat[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
#########################################################################################
###
gen_data = {"n":100000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
SGD_100000_mat = SGD_num_exp(gen_data, 0.0005, 25, 100)
SGD_100000_dat = pd.DataFrame(SGD_100000_mat)
SGD_100000_dat.to_csv('/Users/lizhesun/Documents/Project_One_IncreasementRegression/SGD_result/beta1/SGD_100000_1000_beta001_dat.txt', sep = ',')
DR_exp4 = np.sum(SGD_100000_mat[:,2] == gen_data["k"])
PCD_exp4 = np.mean(SGD_100000_mat[:,2] / gen_data["k"])
rmse_ave_exp4 = np.mean(SGD_100000_mat[:,3])
running_time_ave_exp4 = np.mean(SGD_100000_mat[:,4])
print(DR_exp4)
print(PCD_exp4)
print(rmse_ave_exp4)
print(running_time_ave_exp4)
###
##########################################################################################
###
gen_data = {"n":300000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
SGD_300000_mat = SGD_num_exp(gen_data, 0.0005, 25, 100)
SGD_300000_dat = pd.DataFrame(SGD_300000_mat)
SGD_300000_dat.to_csv('/Users/lizhesun/Documents/Project_One_IncreasementRegression/SGD_result/beta1/SGD_300000_1000_beta001_dat.txt', sep = ',')
DR_exp5 = np.sum(SGD_300000_mat[:,2] == gen_data["k"])
PCD_exp5 = np.mean(SGD_300000_mat[:,2] / gen_data["k"])
rmse_ave_exp5 = np.mean(SGD_300000_mat[:,3])
running_time_ave_exp5 = np.mean(SGD_300000_mat[:,4])
print(DR_exp5)
print(PCD_exp5)
print(rmse_ave_exp5)
print(running_time_ave_exp5)
###
#########################################################################################
###
gen_data = {"n":1000000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
SGD_1000000_mat = SGD_num_exp(gen_data, 0.0005, 0, 25, 100)
SGD_1000000_dat = pd.DataFrame(SGD_1000000_mat)
SGD_1000000_dat.to_csv('/Users/lizhesun/Documents/Project_One_IncreasementRegression/SGD_result/beta1/SGD_1000000_1000_beta001_dat.txt', sep = ',')
DR_exp6 = np.sum(SGD_1000000_mat[:,2] == gen_data["k"])
PCD_exp6 = np.mean(SGD_1000000_mat[:,2] / gen_data["k"])
rmse_ave_exp6 = np.mean(SGD_1000000_mat[:,3])
running_time_ave_exp6 = np.mean(SGD_1000000_mat[:,4])
print(DR_exp6)
print(PCD_exp6)
print(rmse_ave_exp6)
print(running_time_ave_exp6)
###
#########################################################################################
###
############################
### Big data experiment
############################
###############
## beta = 1
###############
###
##############################################################################
###
gen_data = {"n":10000, "p":10000, "k":1000, "alpha":1, "beta_star":1, "dat_type":1}
SGD_10000_mat = SGD_num_exp(gen_data, 0.00005, 25, 20)
SGD_10000_dat = pd.DataFrame(SGD_10000_mat)
SGD_10000_dat.to_csv('/Users/lizhesun/Documents/Project_One_IncreasementRegression/SGD_result/beta1/SGD_10000_10000_dat.txt', sep = ',')
DR_exp1 = np.sum(SGD_10000_mat[:,2] == gen_data["k"])
PCD_exp1 = np.mean(SGD_10000_mat[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(SGD_10000_mat[:,3])
running_time_ave_exp1 = np.mean(SGD_10000_mat[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
###############################################################################
###
gen_data = {"n":30000, "p":10000, "k":1000, "alpha":1, "beta_star":1, "dat_type":1}
SGD_30000_mat = SGD_num_exp(gen_data, 0.00005, 25, 20)
SGD_30000_dat = pd.DataFrame(SGD_30000_mat)
SGD_30000_dat.to_csv('/Users/lizhesun/Documents/Project_One_IncreasementRegression/SGD_result/beta1/SGD_30000_10000_dat.txt', sep = ',')
DR_exp2 = np.sum(SGD_30000_mat[:,2] == gen_data["k"])
PCD_exp2 = np.mean(SGD_30000_mat[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(SGD_30000_mat[:,3])
running_time_ave_exp2 = np.mean(SGD_30000_mat[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
#########################################################################
###
gen_data = {"n":100000, "p":10000, "k":1000, "alpha":1, "beta_star":1, "dat_type":1}
SGD_100000_mat = SGD_num_exp(gen_data, 0.00005, 25, 20)
SGD_100000_dat = pd.DataFrame(SGD_100000_mat)
SGD_100000_dat.to_csv('/Users/lizhesun/Documents/Project_One_IncreasementRegression/SGD_result/beta1/SGD_100000_10000_dat.txt', sep = ',')
DR_exp3 = np.sum(SGD_100000_mat[:,2] == gen_data["k"])
PCD_exp3 = np.mean(SGD_100000_mat[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(SGD_100000_mat[:,3])
running_time_ave_exp3 = np.mean(SGD_100000_mat[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
########################################################################
###
###############
## beta = 0.1
###############
###
##############################################################################
###
gen_data = {"n":10000, "p":10000, "k":1000, "alpha":1, "beta_star":0.1, "dat_type":1}
SGD_10000_mat = SGD_num_exp(gen_data, 0.00005, 25, 20)
SGD_10000_dat = pd.DataFrame(SGD_10000_mat)
SGD_10000_dat.to_csv('/Users/lizhesun/Documents/Project_One_IncreasementRegression/SGD_result/beta1/SGD_10000_10000_beta01_dat.txt', sep = ',')
DR_exp1 = np.sum(SGD_10000_mat[:,2] == gen_data["k"])
PCD_exp1 = np.mean(SGD_10000_mat[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(SGD_10000_mat[:,3])
running_time_ave_exp1 = np.mean(SGD_10000_mat[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
###############################################################################
###
gen_data = {"n":30000, "p":10000, "k":1000, "alpha":1, "beta_star":0.1, "dat_type":1}
SGD_30000_mat = SGD_num_exp(gen_data, 0.00005, 25, 20)
SGD_30000_dat = pd.DataFrame(SGD_30000_mat)
SGD_30000_dat.to_csv('/Users/lizhesun/Documents/Project_One_IncreasementRegression/SGD_result/beta1/SGD_30000_10000_beta01_dat.txt', sep = ',')
DR_exp2 = np.sum(SGD_30000_mat[:,2] == gen_data["k"])
PCD_exp2 = np.mean(SGD_3000_mat[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(SGD_3000_mat[:,3])
running_time_ave_exp2 = np.mean(SGD_3000_mat[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
#########################################################################
###
gen_data = {"n":100000, "p":10000, "k":1000, "alpha":1, "beta_star":0.1, "dat_type":1}
SGD_100000_mat = SGD_num_exp(gen_data, 0.00005, 25, 20)
SGD_100000_dat = pd.DataFrame(SGD_100000_mat)
SGD_100000_dat.to_csv('/Users/lizhesun/Documents/Project_One_IncreasementRegression/SGD_result/beta1/SGD_100000_10000_beta01_dat.txt', sep = ',')
DR_exp3 = np.sum(SGD_100000_mat[:,2] == gen_data["k"])
PCD_exp3 = np.mean(SGD_100000_mat[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(SGD_100000_mat[:,3])
running_time_ave_exp3 = np.mean(SGD_100000_mat[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
#########################################################################
###
###############
## beta = 0.01
###############
#####################
### beta = 0.01
#####################
###
########################################################################################
###
gen_data = {"n":10000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
SGD_10000_mat = SGD_num_exp(gen_data, 0.00005, 0, 25, 20)
SGD_10000_dat = pd.DataFrame(SGD_10000_mat)
SGD_10000_dat.to_csv('/Users/lizhesun/Documents/Project_One_IncreasementRegression/SGD_result/beta1/SGD_10000_10000_beta001_dat.txt', sep = ',')
DR_exp1 = np.sum(SGD_10000_mat[:,2] == gen_data["k"])
PCD_exp1 = np.mean(SGD_10000_mat[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(SGD_10000_mat[:,3])
running_time_ave_exp1 = np.mean(SGD_10000_mat[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
########################################################################################
###
gen_data = {"n":30000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
SGD_30000_mat = SGD_num_exp(gen_data, 0.00005, 0, 25, 20)
SGD_30000_dat = pd.DataFrame(SGD_30000_mat)
SGD_30000_dat.to_csv('/Users/lizhesun/Documents/Project_One_IncreasementRegression/SGD_result/beta1/SGD_30000_10000_beta001_dat.txt', sep = ',')
DR_exp2 = np.sum(SGD_30000_mat[:,2] == gen_data["k"])
PCD_exp2 = np.mean(SGD_30000_mat[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(SGD_30000_mat[:,3])
running_time_ave_exp2 = np.mean(SGD_30000_mat[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
########################################################################################
###
gen_data = {"n":100000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
SGD_100000_mat = SGD_num_exp(gen_data, 0.00005, 0, 25, 20)
SGD_100000_dat = pd.DataFrame(SGD_10000_mat)
SGD_100000_dat.to_csv('/Users/lizhesun/Documents/Project_One_IncreasementRegression/SGD_result/beta1/SGD_100000_10000_beta001_dat.txt', sep = ',')
DR_exp3 = np.sum(SGD_10000_mat[:,2] == gen_data["k"])
PCD_exp3 = np.mean(SGD_10000_mat[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(SGD_10000_mat[:,3])
running_time_ave_exp3 = np.mean(SGD_10000_mat[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
#########################################################################################
###
gen_data = {"n":100000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
SGD_100000_mat = SGD_num_exp(gen_data, 0.00005, 0, 25, 20)
SGD_100000_dat = pd.DataFrame(SGD_100000_mat)
SGD_100000_dat.to_csv('/Users/lizhesun/Documents/Project_One_IncreasementRegression/SGD_result/beta1/SGD_100000_10000_beta001_dat.txt', sep = ',')
DR_exp4 = np.sum(SGD_100000_mat[:,2] == gen_data["k"])
PCD_exp4 = np.mean(SGD_100000_mat[:,2] / gen_data["k"])
rmse_ave_exp4 = np.mean(SGD_100000_mat[:,3])
running_time_ave_exp4 = np.mean(SGD_100000_mat[:,4])
print(DR_exp4)
print(PCD_exp4)
print(rmse_ave_exp4)
print(running_time_ave_exp4)
###
##########################################################################################
###
gen_data = {"n":300000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
SGD_300000_mat = SGD_num_exp(gen_data, 0.00005, 0, 25, 20)
SGD_300000_dat = pd.DataFrame(SGD_300000_mat)
SGD_300000_dat.to_csv('/Users/lizhesun/Documents/Project_One_IncreasementRegression/SGD_result/beta1/SGD_300000_10000_beta001_dat.txt', sep = ',')
DR_exp5 = np.sum(SGD_300000_mat[:,2] == gen_data["k"])
PCD_exp5 = np.mean(SGD_300000_mat[:,2] / gen_data["k"])
rmse_ave_exp5 = np.mean(SGD_300000_mat[:,3])
running_time_ave_exp5 = np.mean(SGD_300000_mat[:,4])
print(DR_exp5)
print(PCD_exp5)
print(rmse_ave_exp5)
print(running_time_ave_exp5)
###
#########################################################################################
###
gen_data = {"n":1000000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
SGD_1000000_mat = SGD_num_exp(gen_data, 0.00005, 0, 25, 20)
SGD_1000000_dat = pd.DataFrame(SGD_1000000_mat)
SGD_1000000_dat.to_csv('/Users/lizhesun/Documents/Project_One_IncreasementRegression/SGD_result/beta1/SGD_1000000_10000_beta001_dat.txt', sep = ',')
DR_exp6 = np.sum(SGD_1000000_mat[:,2] == gen_data["k"])
PCD_exp6 = np.mean(SGD_1000000_mat[:,2] / gen_data["k"])
rmse_ave_exp6 = np.mean(SGD_1000000_mat[:,3])
running_time_ave_exp6 = np.mean(SGD_1000000_mat[:,4])
print(DR_exp6)
print(PCD_exp6)
print(rmse_ave_exp6)
print(running_time_ave_exp6)
###