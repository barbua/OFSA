# -*- coding: utf-8 -*-
######################################
###  OLSth Numerical Experiment (Project1)
######################################
###         Load package
######################################
import onlineFSA
import time
import sys
sys.path.append("/Users/lizhesun/Documents/OFSelection_2023/simulations/classification")
import eqcorrdata
import numpy as np
from sklearn.metrics import roc_curve, auc
########################################################################################
########################################################################################
def OLSth_feasel(gen_data, batch_size, lbd):
    #####################
    #####################
    n = gen_data["n"]
    p = gen_data["p"]
    k = gen_data["k"]
    alpha = gen_data["alpha"]
    beta_star = gen_data["beta_star"]
    ###########################
    ### initial value(running averages)
    ###########################
    num_batch = int(n / batch_size)
    #################
    n_sum = 0
    Sx_sum = np.zeros((1, p))
    Sy_sum = 0
    Sxx_sum = np.zeros((p, p))
    Sxy_sum = np.zeros((p, 1))
    Syy_sum = 0
    ########
    ra_sum = {"n":n_sum, "Sx":Sx_sum, "Sy":Sy_sum, "Sxx":Sxx_sum, "Sxy":Sxy_sum, "Syy":Syy_sum}
    gen_data_batch = {"n":batch_size, "p":p, "k":k, "alpha":alpha, "beta_star":beta_star}
    ########
    ########
    for i in range(num_batch):
        Xtr_mb, Ytr_mb, betastar_vec, istar = eqcorrdata.eqcorrdat_cls(gen_data_batch)
        ra_mb = onlineFSA.running_aves(Xtr_mb, Ytr_mb)
        ra_sum = onlineFSA.add_runningaves(ra_sum, ra_mb)
    #######
    #######
    del Xtr_mb, Ytr_mb
    #######
    if ra_sum["n"] != n:
        print("error")
    #######
    ## Standardize Running averages
    #######
    XX_normalize, XY_normalize, mu_x, mu_y, std_x = onlineFSA.standardize_ra(ra_sum)
    ###
    ### Setup start time
    ###
    t_start = time.process_time()
    ### FSA Experiment
    beta_hat_OLS = onlineFSA.OLS_runningaves(XX_normalize, XY_normalize, 0)
    beta_hat_OLS = beta_hat_OLS.ravel()
    beta_index = np.argsort(abs(beta_hat_OLS))
    beta_index = beta_index.ravel()
    beta_index_de = beta_index[::-1]
    sel = beta_index_de[0:k]
    #################
    t_end = time.process_time()
    time_cost = t_end - t_start
    ### Compare the index with the true index
    num_true = len(np.intersect1d(istar, sel))
    #################
    ######
    ## test data
    ######
    gen_testdata = {"n":batch_size, "p":p, "k":k, "alpha":alpha, "beta_star":beta_star}
    testX, testY, betastar_vec, istar = eqcorrdata.eqcorrdat_cls(gen_testdata)
    ################
    ### Standardize the test data
    ################
    testX_standardize = testX - np.ones((batch_size, 1)).dot(mu_x)
    inv_sigma = 1 / std_x
    testX_standardize = inv_sigma * testX_standardize
    #######
    ### refit by using OLS
    #######
    sel = sel.ravel()
    XX_sel = XX_normalize[np.ix_(sel, sel)]
    XY_sel = XY_normalize[sel]
    beta_sel = onlineFSA.OLS_runningaves(XX_sel, XY_sel, lbd)
    beta_OLSth = np.zeros((p, 1))
    beta_OLSth[sel] = beta_sel
    ######
    ######
    Yscore_test = testX_standardize.dot(beta_OLSth)
    fpr, tpr, thresholds = roc_curve(testY, Yscore_test)
    roc_auc = auc(fpr, tpr)
    ######
    return(num_true, roc_auc, time_cost)
########################################################################################
########################################################################################
########################################################################################
def OLSth_numexp(gen_data, batch_size, lbd, exp_times):
    ####################
    OLSth_results = np.zeros((exp_times, 4))
    ####################
    for i in range(exp_times):
        ######################
        ### seed (classification)
        ######################
        np.random.seed(1991 + i)
        ######################
        num_true, roc_auc, t_feasel = OLSth_feasel(gen_data, batch_size, lbd)
        OLSth_results[i, 0] = 1991 + i
        OLSth_results[i, 1] = num_true
        OLSth_results[i, 2] = roc_auc
        OLSth_results[i, 3] = t_feasel
        #######################
    return(OLSth_results)
########################################################################################
###
########################################################################################
########################################################################################
###
###################
### p = 1000, k = 100, n = 10000, 30000, 100000, beta = 1 
###################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":1}    
OLSth1000_beta1 = OLSth_numexp(gen_data, 1000, 0.01, 100)
DR_exp = np.sum(OLSth1000_beta1[:,1] == gen_data["k"])
PCD_exp = np.mean(OLSth1000_beta1[:,1] / gen_data["k"])
auc_exp = np.mean(OLSth1000_beta1[:,2])
time_exp = np.mean(OLSth1000_beta1[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":1}    
OLSth10000_beta1 = OLSth_numexp(gen_data, 1000, 0.01, 100)
DR_exp = np.sum(OLSth10000_beta1[:,1] == gen_data["k"])
PCD_exp = np.mean(OLSth10000_beta1[:,1] / gen_data["k"])
auc_exp = np.mean(OLSth10000_beta1[:,2])
time_exp = np.mean(OLSth10000_beta1[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":30000, "p":1000, "k":100, "alpha":1, "beta_star":1}    
OLSth30000_beta1 = OLSth_numexp(gen_data, 1000, 0.01, 100)
DR_exp = np.sum(OLSth30000_beta1[:,1] == gen_data["k"])
PCD_exp = np.mean(OLSth30000_beta1[:,1] / gen_data["k"])
auc_exp = np.mean(OLSth30000_beta1[:,2])
time_exp = np.mean(OLSth30000_beta1[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":100000, "p":1000, "k":100, "alpha":1, "beta_star":1}
OLSth100000_beta1 = OLSth_numexp(gen_data, 1000, 0.01, 100)
DR_exp = np.sum(OLSth100000_beta1[:,1] == gen_data["k"])
PCD_exp = np.mean(OLSth100000_beta1[:,1] / gen_data["k"])
auc_exp = np.mean(OLSth100000_beta1[:,2])
time_exp = np.mean(OLSth100000_beta1[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
###
###################
### p = 1000, k = 100, n = 10000, 30000, 100000, beta = 0.01 
###################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":0.01}    
OLSth1000_beta001 = OLSth_numexp(gen_data, 1000, 0.01, 100)
DR_exp = np.sum(OLSth1000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OLSth1000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OLSth1000_beta001[:,2])
time_exp = np.mean(OLSth1000_beta001[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":0.01}    
OLSth10000_beta001 = OLSth_numexp(gen_data, 1000, 0.01, 100)
DR_exp = np.sum(OLSth10000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OLSth10000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OLSth10000_beta001[:,2])
time_exp = np.mean(OLSth10000_beta001[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":30000, "p":1000, "k":100, "alpha":1, "beta_star":0.01}    
OLSth30000_beta001 = OLSth_numexp(gen_data, 1000, 0.01, 100)
DR_exp = np.sum(OLSth30000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OLSth30000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OLSth30000_beta001[:,2])
time_exp = np.mean(OLSth30000_beta001[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":100000, "p":1000, "k":100, "alpha":1, "beta_star":0.01}
OLSth100000_beta001 = OLSth_numexp(gen_data, 1000, 0.01, 100)
DR_exp = np.sum(OLSth100000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OLSth100000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OLSth100000_beta001[:,2])
time_exp = np.mean(OLSth100000_beta001[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":300000, "p":1000, "k":100, "alpha":1, "beta_star":0.01}
OLSth300000_beta001 = OLSth_numexp(gen_data, 1000, 0.01, 100)
DR_exp = np.sum(OLSth300000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OLSth300000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OLSth300000_beta001[:,2])
time_exp = np.mean(OLSth300000_beta001[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":1000000, "p":1000, "k":100, "alpha":1, "beta_star":0.01}
OLSth1000000_beta001 = OLSth_numexp(gen_data, 1000, 0.01, 100)
DR_exp = np.sum(OLSth1000000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OLSth1000000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OLSth1000000_beta001[:,2])
time_exp = np.mean(OLSth1000000_beta001[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)