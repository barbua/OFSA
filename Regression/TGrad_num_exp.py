# -*- coding: utf-8 -*-
##########################################
## Truncate Gradient
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
def TGrad_num_exp(gen_data, lbd_par, eta, minibatch, loop_times):
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
        if n >= p:
            batch = p
        else:
            batch = n
        outer_loop = int(n / batch)
        beta = np.zeros((p, 1))
        beta0 = 0
        time_total = 0
        ###
        for j in range(outer_loop):
            ############
            ### generate data
            ############
            gen_data_bat = {"n":batch, "p":p, "k":k, "alpha":alpha, "beta_star":beta_star, "dat_type":dat_type}
            Xtr_batch, Ytr_batch, betastar_vec, istar = datageneration.generate_data(gen_data_bat)
            #############################
            ###########
            ###lbd
            ###########
            lbd_start = lbd_par["lbd_start"]
            lbd_end = lbd_par["lbd_end"]
            lbd_vec = np.exp(np.linspace(lbd_start, lbd_end, 200))
            ###########
            beta_mat = np.zeros((p, lbd_vec.shape[0]))
            beta0_vec = np.zeros(lbd_vec.shape[0])
            sel_fea_vec = np.zeros(lbd_vec.shape[0])
            ###########
            #############################
            ### TGrad algorithm
            #############################
            t_start = time.process_time()
            for Iter in range(lbd_vec.shape[0]):
                lbd = lbd_vec[Iter]
                beta, beta0 = SGD.train_TGrad(Xtr_batch, Ytr_batch, beta, beta0, eta, lbd, minibatch)
                beta_mat[:,Iter] = beta.ravel() 
                beta0_vec[Iter] = beta0
                sel_fea_vec[Iter] = np.flatnonzero(beta).shape[0]
            t_end = time.process_time()
            time_cost = t_end - t_start
            time_total = time_total + time_cost
        ################################
        result_mat[i, 4] = time_total
        lbd_index = np.where(sel_fea_vec <= k)[0]
        lbd_sel_index = lbd_index[0]
        lbd_select = lbd_vec[lbd_sel_index]
        result_mat[i, 1] = lbd_select
        print(sel_fea_vec[lbd_sel_index], i)
        beta_TGrad = beta_mat[:, lbd_sel_index].reshape(p, 1)
        beta0_TGrad = beta0_vec[lbd_sel_index]
        sel_index = np.flatnonzero(beta_TGrad)
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
        testY_hat = testX.dot(beta_TGrad) + beta0_TGrad * np.ones((batch, 1))
        err_hat = testY.T - testY_hat.T
        rmse = np.sqrt(np.sum(err_hat**2) / batch) 
        result_mat[i, 3] = rmse
        ########
    return(result_mat)
########################################################################################
###
########################################################################################
###
gen_data = {"n":300, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
lbd_par = {"lbd_start":5, "lbd_end":7}
TGrad_300_beta1 = TGrad_num_exp(gen_data, lbd_par, 0.0001, 25, 100)
DR_exp00 = np.sum(TGrad_300_beta1[:,2] == gen_data["k"])
PCD_exp00 = np.mean(TGrad_300_beta1[:,2] / gen_data["k"])
rmse_ave_exp00 = np.mean(TGrad_300_beta1[:,3])
running_time_ave_exp00 = np.mean(TGrad_300_beta1[:,4])
print(DR_exp00)
print(PCD_exp00)
print(rmse_ave_exp00)
print(running_time_ave_exp00)
###
########################################################################################
###
gen_data = {"n":500, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
lbd_par = {"lbd_start":5, "lbd_end":7}
TGrad_500_beta1 = TGrad_num_exp(gen_data, lbd_par, 0.0001, 25, 100)
DR_exp0 = np.sum(TGrad_500_beta1[:,2] == gen_data["k"])
PCD_exp0 = np.mean(TGrad_500_beta1[:,2] / gen_data["k"])
rmse_ave_exp0 = np.mean(TGrad_500_beta1[:,3])
running_time_ave_exp0 = np.mean(TGrad_500_beta1[:,4])
print(DR_exp0)
print(PCD_exp0)
print(rmse_ave_exp0)
print(running_time_ave_exp0)
###
########################################################################################
###beta = 1
########################################################################################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
lbd_par = {"lbd_start":5, "lbd_end":7}
TGrad_1000_beta1 = TGrad_num_exp(gen_data, lbd_par, 0.0001, 25, 100)
DR_exp1 = np.sum(TGrad_1000_beta1[:,2] == gen_data["k"])
PCD_exp1 = np.mean(TGrad_1000_beta1[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(TGrad_1000_beta1[:,3])
running_time_ave_exp1 = np.mean(TGrad_1000_beta1[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
########################################################################################
###
gen_data = {"n":3000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
lbd_par = {"lbd_start":5, "lbd_end":7}
TGrad_3000_beta1 = TGrad_num_exp(gen_data, lbd_par, 0.0001, 25, 100)
DR_exp2 = np.sum(TGrad_3000_beta1[:,2] == gen_data["k"])
PCD_exp2 = np.mean(TGrad_3000_beta1[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(TGrad_3000_beta1[:,3])
running_time_ave_exp2 = np.mean(TGrad_3000_beta1[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
########################################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
lbd_par = {"lbd_start":5, "lbd_end":7}
TGrad_10000_beta1 = TGrad_num_exp(gen_data, lbd_par, 0.0001, 25, 100)
DR_exp3 = np.sum(TGrad_10000_beta1[:,2] == gen_data["k"])
PCD_exp3 = np.mean(TGrad_10000_beta1[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(TGrad_10000_beta1[:,3])
running_time_ave_exp3 = np.mean(TGrad_10000_beta1[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
########################################################################################
###
########################################################################################
### beta = 0.1
########################################################################################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":0.1, "dat_type":1}
lbd_par = {"lbd_start":3, "lbd_end":5}
TGrad_1000_beta01 = TGrad_num_exp(gen_data, lbd_par, 0.0001, 25, 100)
DR_exp1 = np.sum(TGrad_1000_beta01[:,2] == gen_data["k"])
PCD_exp1 = np.mean(TGrad_1000_beta01[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(TGrad_1000_beta01[:,3])
running_time_ave_exp1 = np.mean(TGrad_1000_beta01[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
########################################################################################
###
gen_data = {"n":3000, "p":1000, "k":100, "alpha":1, "beta_star":0.1, "dat_type":1}
lbd_par = {"lbd_start":3, "lbd_end":5}
TGrad_3000_beta01 = TGrad_num_exp(gen_data, lbd_par, 0.0001, 25, 100)
DR_exp2 = np.sum(TGrad_3000_beta01[:,2] == gen_data["k"])
PCD_exp2 = np.mean(TGrad_3000_beta01[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(TGrad_3000_beta01[:,3])
running_time_ave_exp2 = np.mean(TGrad_3000_beta01[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
########################################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":0.1, "dat_type":1}
lbd_par = {"lbd_start":3, "lbd_end":5}
TGrad_10000_beta01 = TGrad_num_exp(gen_data, lbd_par, 0.0001, 25, 100)
DR_exp3 = np.sum(TGrad_10000_beta01[:,2] == gen_data["k"])
PCD_exp3 = np.mean(TGrad_10000_beta01[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(TGrad_10000_beta01[:,3])
running_time_ave_exp3 = np.mean(TGrad_10000_beta01[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
########################################################################################
###
###############
###beta = 0.01
###############
###
gen_data = {"n":500, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_par = {"lbd_start":1, "lbd_end":3}
TGrad_500_beta001 = TGrad_num_exp(gen_data, lbd_par, 0.0001, 25, 100)
DR_exp0 = np.sum(TGrad_500_beta001[:,2] == gen_data["k"])
PCD_exp0 = np.mean(TGrad_500_beta001[:,2] / gen_data["k"])
rmse_ave_exp0 = np.mean(TGrad_500_beta001[:,3])
running_time_ave_exp0 = np.mean(TGrad_500_beta001[:,4])
print(DR_exp0)
print(PCD_exp0)
print(rmse_ave_exp0)
print(running_time_ave_exp0)
###
########################################################################################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_par = {"lbd_start":1, "lbd_end":3}
TGrad_1000_beta001 = TGrad_num_exp(gen_data, lbd_par, 0.0001, 25, 100)
DR_exp1 = np.sum(TGrad_1000_beta001[:,2] == gen_data["k"])
PCD_exp1 = np.mean(TGrad_1000_beta001[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(TGrad_1000_beta001[:,3])
running_time_ave_exp1 = np.mean(TGrad_1000_beta001[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
########################################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_par = {"lbd_start":1, "lbd_end":3}
TGrad_10000_beta001 = TGrad_num_exp(gen_data, lbd_par, 0.0001, 25, 100)
DR_exp2 = np.sum(TGrad_10000_beta001[:,2] == gen_data["k"])
PCD_exp2 = np.mean(TGrad_10000_beta001[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(TGrad_10000_beta001[:,3])
running_time_ave_exp2 = np.mean(TGrad_10000_beta001[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
########################################################################################
###
gen_data = {"n":100000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_par = {"lbd_start":1, "lbd_end":3}
TGrad_100000_beta001 = TGrad_num_exp(gen_data, lbd_par, 0.0001, 25, 100)
DR_exp3 = np.sum(TGrad_100000_beta001[:,2] == gen_data["k"])
PCD_exp3 = np.mean(TGrad_100000_beta001[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(TGrad_100000_beta001[:,3])
running_time_ave_exp3 = np.mean(TGrad_100000_beta001[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
########################################################################################
###
gen_data = {"n":300000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_par = {"lbd_start":1, "lbd_end":3}
TGrad_300000_beta001 = TGrad_num_exp(gen_data, lbd_par, 0.0001, 25, 100)
DR_exp4 = np.sum(TGrad_300000_beta001[:,2] == gen_data["k"])
PCD_exp4 = np.mean(TGrad_300000_beta001[:,2] / gen_data["k"])
rmse_ave_exp4 = np.mean(TGrad_300000_beta001[:,3])
running_time_ave_exp4 = np.mean(TGrad_300000_beta001[:,4])
print(DR_exp4)
print(PCD_exp4)
print(rmse_ave_exp4)
print(running_time_ave_exp4)
###
########################################################################################
###
gen_data = {"n":1000000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_par = {"lbd_start":1, "lbd_end":3}
TGrad_1000000_beta001 = TGrad_num_exp(gen_data, lbd_par, 0.0001, 25, 100)
DR_exp5 = np.sum(TGrad_1000000_beta001[:,2] == gen_data["k"])
PCD_exp5 = np.mean(TGrad_1000000_beta001[:,2] / gen_data["k"])
rmse_ave_exp5 = np.mean(TGrad_1000000_beta001[:,3])
running_time_ave_exp5 = np.mean(TGrad_1000000_beta001[:,4])
print(DR_exp5)
print(PCD_exp5)
print(rmse_ave_exp5)
print(running_time_ave_exp5)
###
########################################################################################
###
##################
### Big data experiment
##################
###
########################################################################################
###beta = 1
########################################################################################
###
gen_data = {"n":10000, "p":10000, "k":1000, "alpha":1, "beta_star":1, "dat_type":1}
lbd_par = {"lbd_start":5, "lbd_end":7}
TGrad_10000_beta1 = TGrad_num_exp(gen_data, lbd_par, 0.0001, 25, 20)
DR_exp1 = np.sum(TGrad_10000_beta1[:,2] == gen_data["k"])
PCD_exp1 = np.mean(TGrad_10000_beta1[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(TGrad_10000_beta1[:,3])
running_time_ave_exp1 = np.mean(TGrad_10000_beta1[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
########################################################################################
###
gen_data = {"n":30000, "p":10000, "k":1000, "alpha":1, "beta_star":1, "dat_type":1}
lbd_par = {"lbd_start":5, "lbd_end":7}
TGrad_30000_beta1 = TGrad_num_exp(gen_data, lbd_par, 0.0001, 25, 20)
DR_exp2 = np.sum(TGrad_30000_beta1[:,2] == gen_data["k"])
PCD_exp2 = np.mean(TGrad_30000_beta1[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(TGrad_30000_beta1[:,3])
running_time_ave_exp2 = np.mean(TGrad_30000_beta1[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
########################################################################################
###
gen_data = {"n":100000, "p":10000, "k":1000, "alpha":1, "beta_star":1, "dat_type":1}
lbd_par = {"lbd_start":5, "lbd_end":7}
TGrad_100000_beta1 = TGrad_num_exp(gen_data, lbd_par, 0.0001, 25, 20)
DR_exp3 = np.sum(TGrad_100000_beta1[:,2] == gen_data["k"])
PCD_exp3 = np.mean(TGrad_100000_beta1[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(TGrad_100000_beta1[:,3])
running_time_ave_exp3 = np.mean(TGrad_100000_beta1[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
########################################################################################
###
###
########################################################################################
###beta = 0.1
########################################################################################
###
###
########################################################################################
###
gen_data = {"n":10000, "p":10000, "k":1000, "alpha":1, "beta_star":0.1, "dat_type":1}
lbd_par = {"lbd_start":5, "lbd_end":7}
TGrad_10000_beta01 = TGrad_num_exp(gen_data, lbd_par, 0.0001, 25, 20)
DR_exp1 = np.sum(TGrad_10000_beta01[:,2] == gen_data["k"])
PCD_exp1 = np.mean(TGrad_10000_beta01[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(TGrad_10000_beta01[:,3])
running_time_ave_exp1 = np.mean(TGrad_10000_beta01[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
########################################################################################
###
gen_data = {"n":30000, "p":10000, "k":1000, "alpha":1, "beta_star":0.1, "dat_type":1}
lbd_par = {"lbd_start":5, "lbd_end":7}
TGrad_30000_beta01 = TGrad_num_exp(gen_data, lbd_par, 0.0001, 25, 20)
DR_exp2 = np.sum(TGrad_30000_beta01[:,2] == gen_data["k"])
PCD_exp2 = np.mean(TGrad_30000_beta01[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(TGrad_30000_beta01[:,3])
running_time_ave_exp2 = np.mean(TGrad_30000_beta01[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
########################################################################################
###
gen_data = {"n":100000, "p":10000, "k":1000, "alpha":1, "beta_star":0.1, "dat_type":1}
lbd_par = {"lbd_start":5, "lbd_end":7}
TGrad_100000_beta01 = TGrad_num_exp(gen_data, lbd_par, 0.0001, 25, 20)
DR_exp3 = np.sum(TGrad_100000_beta01[:,2] == gen_data["k"])
PCD_exp3 = np.mean(TGrad_100000_beta01[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(TGrad_100000_beta01[:,3])
running_time_ave_exp3 = np.mean(TGrad_100000_beta01[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
########################################################################################
###
###
########################################################################################
###beta = 0.01
########################################################################################
###
###
########################################################################################
###
gen_data = {"n":10000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_par = {"lbd_start":5, "lbd_end":7}
TGrad_10000_beta001 = TGrad_num_exp(gen_data, lbd_par, 0.0001, 25, 20)
DR_exp1 = np.sum(TGrad_10000_beta001[:,2] == gen_data["k"])
PCD_exp1 = np.mean(TGrad_10000_beta001[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(TGrad_10000_beta001[:,3])
running_time_ave_exp1 = np.mean(TGrad_10000_beta001[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
########################################################################################
###
gen_data = {"n":30000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_par = {"lbd_start":5, "lbd_end":7}
TGrad_30000_beta001 = TGrad_num_exp(gen_data, lbd_par, 0.0001, 25, 20)
DR_exp2 = np.sum(TGrad_30000_beta001[:,2] == gen_data["k"])
PCD_exp2 = np.mean(TGrad_30000_beta001[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(TGrad_30000_beta001[:,3])
running_time_ave_exp2 = np.mean(TGrad_30000_beta001[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
########################################################################################
###
gen_data = {"n":100000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_par = {"lbd_start":5, "lbd_end":7}
TGrad_100000_beta001 = TGrad_num_exp(gen_data, lbd_par, 0.0001, 25, 20)
DR_exp3 = np.sum(TGrad_100000_beta001[:,2] == gen_data["k"])
PCD_exp3 = np.mean(TGrad_100000_beta001[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(TGrad_100000_beta001[:,3])
running_time_ave_exp3 = np.mean(TGrad_100000_beta001[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
########################################################################################
###
gen_data = {"n":300000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_par = {"lbd_start":5, "lbd_end":7}
TGrad_300000_beta001 = TGrad_num_exp(gen_data, lbd_par, 0.0001, 25, 20)
DR_exp4 = np.sum(TGrad_300000_beta001[:,2] == gen_data["k"])
PCD_exp4 = np.mean(TGrad_300000_beta001[:,2] / gen_data["k"])
rmse_ave_exp4 = np.mean(TGrad_300000_beta001[:,3])
running_time_ave_exp4 = np.mean(TGrad_300000_beta001[:,4])
print(DR_exp4)
print(PCD_exp4)
print(rmse_ave_exp4)
print(running_time_ave_exp4)
###
########################################################################################
###
gen_data = {"n":1000000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_par = {"lbd_start":5, "lbd_end":7}
TGrad_1000000_beta001 = TGrad_num_exp(gen_data, lbd_par, 0.0001, 25, 20)
DR_exp5 = np.sum(TGrad_1000000_beta001[:,2] == gen_data["k"])
PCD_exp5 = np.mean(TGrad_1000000_beta001[:,2] / gen_data["k"])
rmse_ave_exp5 = np.mean(TGrad_1000000_beta001[:,3])
running_time_ave_exp5 = np.mean(TGrad_1000000_beta001[:,4])
print(DR_exp5)
print(PCD_exp5)
print(rmse_ave_exp5)
print(running_time_ave_exp5)