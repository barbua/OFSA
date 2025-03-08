# -*- coding: utf-8 -*-
###########################################
### Lasso numerical experiment
### Lizhe Sun
###########################################
##################
### Load package
##################
import time
import numpy as np
import sys
sys.path.append("/Users/lizhesun/Documents/OFSelection_2023/simulations/regression")
import Lassofeaturesel
import datageneration 
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
##############################################################################
### Lasso Numerical Experiment
##############################################################################
def Lasso_num_exp(gen_dat, lbd_para, loop_time):
    ###
    ###
    p = gen_dat["p"]
    k = gen_dat["k"]
    alpha = gen_dat["alpha"]
    beta_star = gen_dat["beta_star"]
    ### Lasso result matrix
    result_mat = np.zeros((loop_time, 5))
    ###
    ###
    ###
    for i in range(loop_time):
        #####
        ### Generate the seed
        np.random.seed(i + 100)
        result_mat[i, 0] = i + 100
        ########
        ### Generate data
        ########
        X_tr, Y_tr, betastar_vec, istar = datageneration.generate_data(gen_dat)
        ## Standardize data
        X_tr_scale = preprocessing.scale(X_tr)
        Y_tr_center = preprocessing.scale(Y_tr, with_std = False)
        #####
        ## implement Lasso
        #####
        t_start = time.process_time()
        lasso_sel_index, lbd_sel = Lassofeaturesel.Lasso_feature_sel(X_tr_scale, Y_tr_center, lbd_para, k)
        t_end = time.process_time()
        time_cost = t_end - t_start
        num_true_var = len(np.intersect1d(istar, lasso_sel_index))
        result_mat[i, 1] = lbd_sel
        result_mat[i, 4] = time_cost
        result_mat[i, 2] = num_true_var
        ###
        ### refit OLS Model
        ###
        X_tr_sel = X_tr_scale[:, lasso_sel_index]
        OLS_fit = LinearRegression(fit_intercept = False).fit(X_tr_sel, Y_tr_center)
        beta_hat_ols = OLS_fit.coef_
        beta_hat_ols = beta_hat_ols.T
        ###
        ### Generate data
        ###
        gen_data2 = {"n":p, "p":p, "k":k, "alpha":alpha, "beta_star":beta_star, "dat_type":1}
        X_test, Y_test, betastar_vec, istar = datageneration.generate_data(gen_data2)
        ###
        ### Standardize the trainingtest data
        ###
        X_scaler = preprocessing.StandardScaler().fit(X_tr)
        Y_scaler = np.mean(Y_tr, axis = 0)
        ###
        ### Use the mean and variance to standardize the test data
        ###
        testX_scale = X_scaler.transform(X_test)
        testX_scale = testX_scale[:, lasso_sel_index]
        testY_center = Y_test.T - Y_scaler
        testY_center = testY_center.T
        ###
        ### RMSE
        ###
        testY_hat = testX_scale.dot(beta_hat_ols)
        err_hat = testY_center.T - testY_hat.T
        rmse = np.sqrt(np.sum(err_hat**2) / p) 
        result_mat[i, 3] = rmse
        ####
        ####
    return(result_mat)


######################
######### beta = 1
######################
###
########################################################################################
###
gen_data = {"n":300, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
lbd_para = {"start":-10, "end":10}
lasso_300_beta1 = Lasso_num_exp(gen_data, lbd_para, 100)
DR_exp0 = np.sum(lasso_300_beta1[:,2] == gen_data["k"])
PCD_exp0 = np.mean(lasso_300_beta1[:,2] / gen_data["k"])
rmse_ave_exp0 = np.mean(lasso_300_beta1[:,3])
running_time_ave_exp0 = np.mean(lasso_300_beta1[:,4])
print(DR_exp0)
print(PCD_exp0)
print(rmse_ave_exp0)
print(running_time_ave_exp0)
###
########################################################################################
###
gen_data = {"n":500, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
lbd_para = {"start":-10, "end":10}
lasso_500_beta1 = Lasso_num_exp(gen_data, lbd_para, 100)
DR_exp1 = np.sum(lasso_500_beta1[:,2] == gen_data["k"])
PCD_exp1 = np.mean(lasso_500_beta1[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(lasso_500_beta1[:,3])
running_time_ave_exp1 = np.mean(lasso_500_beta1[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
########################################################################################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
lbd_para = {"start":-10, "end":10}
lasso_1000_beta1 = Lasso_num_exp(gen_data, lbd_para, 100)
DR_exp2 = np.sum(lasso_1000_beta1[:,2] == gen_data["k"])
PCD_exp2 = np.mean(lasso_1000_beta1[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(lasso_1000_beta1[:,3])
running_time_ave_exp2 = np.mean(lasso_1000_beta1[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
########################################################################################
###
gen_data = {"n":3000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
lbd_para = {"start":-10, "end":10}
lasso_3000_beta1 = Lasso_num_exp(gen_data, lbd_para, 100)
DR_exp3 = np.sum(lasso_3000_beta1[:,2] == gen_data["k"])
PCD_exp3 = np.mean(lasso_3000_beta1[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(lasso_3000_beta1[:,3])
running_time_ave_exp3 = np.mean(lasso_3000_beta1[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
########################################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}
lbd_para = {"start":-10, "end":10}
lasso_10000_beta1 = Lasso_num_exp(gen_data, lbd_para, 100)
DR_exp4 = np.sum(lasso_10000_beta1[:,2] == gen_data["k"])
PCD_exp4 = np.mean(lasso_10000_beta1[:,2] / gen_data["k"])
rmse_ave_exp4 = np.mean(lasso_10000_beta1[:,3])
running_time_ave_exp4 = np.mean(lasso_10000_beta1[:,4])
print(DR_exp4)
print(PCD_exp4)
print(rmse_ave_exp4)
print(running_time_ave_exp4)
###
########################################################################################
###
######################
######### beta = 0.1
######################
#gen_data = {"n":300, "p":1000, "k":100, "alpha":1, "beta_star":0.1, "dat_type":1}
#lbd_para = {"start":-10, "end":10}
#lasso_300_beta01 = Lasso_num_exp(gen_data, lbd_para, 100)
#DR_exp1 = np.sum(lasso_300_beta01[:,2] == gen_data["k"])
#PCD_exp1 = np.mean(lasso_300_beta01[:,2] / gen_data["k"])
#rmse_ave_exp1 = np.mean(lasso_300_beta01[:,3])
#running_time_ave_exp1 = np.mean(lasso_300_beta01[:,4])
#print(DR_exp1)
#print(PCD_exp1)
#print(rmse_ave_exp1)
#print(running_time_ave_exp1)
###
########################################################################################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":0.1, "dat_type":1}
lbd_para = {"start":-10, "end":10}
lasso_1000_beta01 = Lasso_num_exp(gen_data, lbd_para, 100)
DR_exp2 = np.sum(lasso_1000_beta01[:,2] == gen_data["k"])
PCD_exp2 = np.mean(lasso_1000_beta01[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(lasso_1000_beta01[:,3])
running_time_ave_exp2 = np.mean(lasso_1000_beta01[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
########################################################################################
###
gen_data = {"n":3000, "p":1000, "k":100, "alpha":1, "beta_star":0.1, "dat_type":1}
lbd_para = {"start":-10, "end":10}
lasso_3000_beta01 = Lasso_num_exp(gen_data, lbd_para, 100)
DR_exp3 = np.sum(lasso_3000_beta01[:,2] == gen_data["k"])
PCD_exp3 = np.mean(lasso_3000_beta01[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(lasso_3000_beta01[:,3])
running_time_ave_exp3 = np.mean(lasso_3000_beta01[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
########################################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":0.1, "dat_type":1}
lbd_para = {"start":-10, "end":10}
lasso_10000_beta01 = Lasso_num_exp(gen_data, lbd_para, 100)
DR_exp4 = np.sum(lasso_10000_beta01[:,2] == gen_data["k"])
PCD_exp4 = np.mean(lasso_10000_beta01[:,2] / gen_data["k"])
rmse_ave_exp4 = np.mean(lasso_10000_beta01[:,3])
running_time_ave_exp4 = np.mean(lasso_10000_beta01[:,4])
print(DR_exp4)
print(PCD_exp4)
print(rmse_ave_exp4)
print(running_time_ave_exp4)
###
########################################################################################
###
#################
### Beta = 0.01
#################
###
gen_data = {"n":500, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_para = {"start":-10, "end":10}
lasso_500_beta001 = Lasso_num_exp(gen_data, lbd_para, 100)
DR_exp0 = np.sum(lasso_500_beta001[:,2] == gen_data["k"])
PCD_exp0 = np.mean(lasso_500_beta001[:,2] / gen_data["k"])
rmse_ave_exp0 = np.mean(lasso_500_beta001[:,3])
running_time_ave_exp0 = np.mean(lasso_500_beta001[:,4])
print(DR_exp0)
print(PCD_exp0)
print(rmse_ave_exp0)
print(running_time_ave_exp0)
###
########################################################################################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_para = {"start":-10, "end":10}
lasso_1000_beta001 = Lasso_num_exp(gen_data, lbd_para, 100)
DR_exp1 = np.sum(lasso_1000_beta001[:,2] == gen_data["k"])
PCD_exp1 = np.mean(lasso_1000_beta001[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(lasso_1000_beta001[:,3])
running_time_ave_exp1 = np.mean(lasso_1000_beta001[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
########################################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_para = {"start":-10, "end":10}
lasso_10000_beta001 = Lasso_num_exp(gen_data, lbd_para, 100)
DR_exp2 = np.sum(lasso_10000_beta001[:,2] == gen_data["k"])
PCD_exp2 = np.mean(lasso_10000_beta001[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(lasso_10000_beta001[:,3])
running_time_ave_exp2 = np.mean(lasso_10000_beta001[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
########################################################################################
###
gen_data = {"n":100000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_para = {"start":-10, "end":10}
lasso_100000_beta001 = Lasso_num_exp(gen_data, lbd_para, 100)
DR_exp3 = np.sum(lasso_100000_beta001[:,2] == gen_data["k"])
PCD_exp3 = np.mean(lasso_100000_beta001[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(lasso_100000_beta001[:,3])
running_time_ave_exp3 = np.mean(lasso_100000_beta001[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
########################################################################################
###
gen_data = {"n":300000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_para = {"start":-10, "end":10}
lasso_300000_beta001 = Lasso_num_exp(gen_data, lbd_para, 100)
DR_exp4 = np.sum(lasso_300000_beta001[:,2] == gen_data["k"])
PCD_exp4 = np.mean(lasso_300000_beta001[:,2] / gen_data["k"])
rmse_ave_exp4 = np.mean(lasso_300000_beta001[:,3])
running_time_ave_exp4 = np.mean(lasso_300000_beta001[:,4])
print(DR_exp4)
print(PCD_exp4)
print(rmse_ave_exp4)
print(running_time_ave_exp4)
###
########################################################################################
###
gen_data = {"n":1000000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_para = {"start":-10, "end":10}
lasso_1000000_beta001 = Lasso_num_exp(gen_data, lbd_para, 100)
DR_exp5 = np.sum(lasso_1000000_beta001[:,2] == gen_data["k"])
PCD_exp5 = np.mean(lasso_1000000_beta001[:,2] / gen_data["k"])
rmse_ave_exp5 = np.mean(lasso_1000000_beta001[:,3])
running_time_ave_exp5 = np.mean(lasso_1000000_beta001[:,4])
print(DR_exp5)
print(PCD_exp5)
print(rmse_ave_exp5)
print(running_time_ave_exp5)
########################################################################################
######################
## Big Data
######################
######### beta = 1
######################
#gen_data = {"n":3000, "p":10000, "k":1000, "alpha":1, "beta_star":1, "dat_type":1}
#lbd_para = {"start":-10, "end":10}
#lasso_3000_beta1 = Lasso_num_exp(gen_data, lbd_para, 20)
#DR_exp1 = np.sum(lasso_3000_beta1[:,2] == gen_data["k"])
#PCD_exp1 = np.mean(lasso_3000_beta1[:,2] / gen_data["k"])
#rmse_ave_exp1 = np.mean(lasso_3000_beta1[:,3])
#running_time_ave_exp1 = np.mean(lasso_3000_beta1[:,4])
#print(DR_exp1)
#print(PCD_exp1)
#print(rmse_ave_exp1)
#print(running_time_ave_exp1)
###
########################################################################################
###
gen_data = {"n":10000, "p":10000, "k":1000, "alpha":1, "beta_star":1, "dat_type":1}
lbd_para = {"start":-10, "end":10}
lasso_10000_beta1 = Lasso_num_exp(gen_data, lbd_para, 20)
DR_exp2 = np.sum(lasso_10000_beta1[:,2] == gen_data["k"])
PCD_exp2 = np.mean(lasso_10000_beta1[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(lasso_10000_beta1[:,3])
running_time_ave_exp2 = np.mean(lasso_10000_beta1[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
########################################################################################
###
gen_data = {"n":30000, "p":10000, "k":1000, "alpha":1, "beta_star":1, "dat_type":1}
lbd_para = {"start":-10, "end":10}
lasso_30000_beta1 = Lasso_num_exp(gen_data, lbd_para, 20)
DR_exp3 = np.sum(lasso_30000_beta1[:,2] == gen_data["k"])
PCD_exp3 = np.mean(lasso_30000_beta1[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(lasso_30000_beta1[:,3])
running_time_ave_exp3 = np.mean(lasso_30000_beta1[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
########################################################################################
###
gen_data = {"n":100000, "p":10000, "k":1000, "alpha":1, "beta_star":1, "dat_type":1}
lbd_para = {"start":-10, "end":10}
lasso_100000_beta1 = Lasso_num_exp(gen_data, lbd_para, 20)
DR_exp4 = np.sum(lasso_100000_beta1[:,2] == gen_data["k"])
PCD_exp4 = np.mean(lasso_100000_beta1[:,2] / gen_data["k"])
rmse_ave_exp4 = np.mean(lasso_100000_beta1[:,3])
running_time_ave_exp4 = np.mean(lasso_100000_beta1[:,4])
print(DR_exp4)
print(PCD_exp4)
print(rmse_ave_exp4)
print(running_time_ave_exp4)
###
######################
######### beta = 0.1
######################
###
########################################################################################
###
#gen_data = {"n":3000, "p":10000, "k":1000, "alpha":1, "beta_star":0.1, "dat_type":1}
#lbd_para = {"start":-10, "end":10}
#lasso_3000_beta01 = Lasso_num_exp(gen_data, lbd_para, 20)
#DR_exp1 = np.sum(lasso_3000_beta01[:,2] == gen_data["k"])
#PCD_exp1 = np.mean(lasso_3000_beta01[:,2] / gen_data["k"])
#rmse_ave_exp1 = np.mean(lasso_3000_beta01[:,3])
#running_time_ave_exp1 = np.mean(lasso_3000_beta01[:,4])
#print(DR_exp1)
#print(PCD_exp1)
#print(rmse_ave_exp1)
#print(running_time_ave_exp1)
###
########################################################################################
###
gen_data = {"n":10000, "p":10000, "k":1000, "alpha":1, "beta_star":0.1, "dat_type":1}
lbd_para = {"start":-10, "end":10}
lasso_10000_beta01 = Lasso_num_exp(gen_data, lbd_para, 20)
DR_exp2 = np.sum(lasso_10000_beta01[:,2] == gen_data["k"])
PCD_exp2 = np.mean(lasso_10000_beta01[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(lasso_10000_beta01[:,3])
running_time_ave_exp2 = np.mean(lasso_10000_beta01[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
########################################################################################
###
gen_data = {"n":30000, "p":10000, "k":1000, "alpha":1, "beta_star":0.1, "dat_type":1}
lbd_para = {"start":-10, "end":10}
lasso_30000_beta01 = Lasso_num_exp(gen_data, lbd_para, 20)
DR_exp3 = np.sum(lasso_30000_beta01[:,2] == gen_data["k"])
PCD_exp3 = np.mean(lasso_30000_beta01[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(lasso_30000_beta01[:,3])
running_time_ave_exp3 = np.mean(lasso_30000_beta01[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
########################################################################################
###
gen_data = {"n":100000, "p":10000, "k":1000, "alpha":1, "beta_star":1, "dat_type":1}
lbd_para = {"start":-10, "end":10}
lasso_100000_beta01 = Lasso_num_exp(gen_data, lbd_para, 20)
DR_exp4 = np.sum(lasso_100000_beta01[:,2] == gen_data["k"])
PCD_exp4 = np.mean(lasso_100000_beta01[:,2] / gen_data["k"])
rmse_ave_exp4 = np.mean(lasso_100000_beta01[:,3])
running_time_ave_exp4 = np.mean(lasso_100000_beta01[:,4])
print(DR_exp4)
print(PCD_exp4)
print(rmse_ave_exp4)
print(running_time_ave_exp4)
###
########################################################################################
###
########################################################################################
### Beta = 0.01
########################################################################################
###
gen_data = {"n":10000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_para = {"start":-10, "end":10}
lasso_10000_beta001 = Lasso_num_exp(gen_data, lbd_para, 20)
DR_exp1 = np.sum(lasso_10000_beta001[:,2] == gen_data["k"])
PCD_exp1 = np.mean(lasso_10000_beta001[:,2] / gen_data["k"])
rmse_ave_exp1 = np.mean(lasso_10000_beta001[:,3])
running_time_ave_exp1 = np.mean(lasso_10000_beta001[:,4])
print(DR_exp1)
print(PCD_exp1)
print(rmse_ave_exp1)
print(running_time_ave_exp1)
###
########################################################################################
###
gen_data = {"n":30000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_para = {"start":-10, "end":10}
lasso_30000_beta001 = Lasso_num_exp(gen_data, lbd_para, 20)
DR_exp2 = np.sum(lasso_30000_beta001[:,2] == gen_data["k"])
PCD_exp2 = np.mean(lasso_30000_beta001[:,2] / gen_data["k"])
rmse_ave_exp2 = np.mean(lasso_30000_beta001[:,3])
running_time_ave_exp2 = np.mean(lasso_30000_beta001[:,4])
print(DR_exp2)
print(PCD_exp2)
print(rmse_ave_exp2)
print(running_time_ave_exp2)
###
########################################################################################
###
gen_data = {"n":100000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_para = {"start":-6, "end":6}
lasso_100000_beta001 = Lasso_num_exp(gen_data, lbd_para, 20)
DR_exp3 = np.sum(lasso_100000_beta001[:,2] == gen_data["k"])
PCD_exp3 = np.mean(lasso_100000_beta001[:,2] / gen_data["k"])
rmse_ave_exp3 = np.mean(lasso_100000_beta001[:,3])
running_time_ave_exp3 = np.mean(lasso_100000_beta001[:,4])
print(DR_exp3)
print(PCD_exp3)
print(rmse_ave_exp3)
print(running_time_ave_exp3)
###
########################################################################################
###
gen_data = {"n":300000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_para = {"start":-10, "end":10}
lasso_300000_beta001 = Lasso_num_exp(gen_data, lbd_para, 20)
DR_exp4 = np.sum(lasso_300000_beta001[:,2] == gen_data["k"])
PCD_exp4 = np.mean(lasso_300000_beta001[:,2] / gen_data["k"])
rmse_ave_exp4 = np.mean(lasso_300000_beta001[:,3])
running_time_ave_exp4 = np.mean(lasso_300000_beta001[:,4])
print(DR_exp4)
print(PCD_exp4)
print(rmse_ave_exp4)
print(running_time_ave_exp4)
###
########################################################################################
###
gen_data = {"n":1000000, "p":10000, "k":1000, "alpha":1, "beta_star":0.01, "dat_type":1}
lbd_para = {"start":-10, "end":10}
lasso_1000000_beta001 = Lasso_num_exp(gen_data, lbd_para, 20)
DR_exp5 = np.sum(lasso_1000000_beta001[:,2] == gen_data["k"])
PCD_exp5 = np.mean(lasso_1000000_beta001[:,2] / gen_data["k"])
rmse_ave_exp5 = np.mean(lasso_1000000_beta001[:,3])
running_time_ave_exp5 = np.mean(lasso_1000000_beta001[:,4])
print(DR_exp5)
print(PCD_exp5)
print(rmse_ave_exp5)
print(running_time_ave_exp5)







