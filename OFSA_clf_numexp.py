# -*- coding: utf-8 -*-
######################################
###  OFSA Numerical Experiment (Project1)
######################################
###         Load package
######################################
import onlineFSA
import sys
sys.path.append("/Users/lizhesun/Documents/OFSelection_2023/simulations/classification")
import time
import eqcorrdata
import numpy as np
from sklearn.metrics import roc_curve, auc
########################################################################################
########################################################################################
def OFSA_feasel(gen_data, OFSA_para, batch_size):
    #####################
    #####################
    n = gen_data["n"]
    p = gen_data["p"]
    k = gen_data["k"]
    alpha = gen_data["alpha"]
    beta_star = gen_data["beta_star"]
    ###################
    ### FSA_para
    ###################
    eta = OFSA_para["eta"]
    mu = OFSA_para["mu"]
    lbd = OFSA_para["lbd"]
    N_iter = OFSA_para["N_iter"]
    pretr_time = OFSA_para["pretr_time"]
    ###########################
    ### initial value (running averages)
    ###########################
    num_batch = int(n / batch_size)
    ###########################
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
    for i in range(num_batch):
        Xtr_mb, Ytr_mb, betastar_vec, istar = eqcorrdata.eqcorrdat_cls(gen_data_batch)
        ra_mb = onlineFSA.running_aves(Xtr_mb, Ytr_mb)
        ra_sum = onlineFSA.add_runningaves(ra_sum, ra_mb)
    #######
    del Xtr_mb, Ytr_mb
    #######
    if ra_sum["n"] != n:
        print("error")
    #######
    ## Standardize Running averages
    #######
    XX_normalize, XY_normalize, mu_x, mu_y, std_x = onlineFSA.standardize_ra(ra_sum)
    ##
    ####################################################################################
    ##
    FSA_para = {"n":n, "k":k, "eta":eta, "mu":mu, "lbd":lbd, "N_iter":N_iter}
    ## Setup start time
    t_start = time.process_time()
    ## FSA Experiment
    beta_sel, sel = onlineFSA.onlineFSA(XX_normalize, XY_normalize, FSA_para, pretr_time)
    #################
    t_end = time.process_time()
    time_cost = t_end - t_start
    ## Compare the index with the true index
    num_true = len(np.intersect1d(istar, sel))
    #################
    ##########
    ### Test data    
    ##########
    testdat_para = {"n":batch_size, "p":p, "k":k, "alpha":alpha, "beta_star":beta_star}
    Xtest, Ytest, betastar_vec, istar = eqcorrdata.eqcorrdat_cls(testdat_para)
    ################
    ### Standardize the test data
    ################
    Xtest_standardize = Xtest - np.ones((batch_size, 1)).dot(mu_x)
    inv_sigma = 1 / std_x
    Xtest_standardize = inv_sigma * Xtest_standardize
    beta_OFSA = np.zeros((p, 1))
    beta_OFSA[sel] = beta_sel
    Yscore_test = Xtest_standardize.dot(beta_OFSA)
    fpr, tpr, thresholds = roc_curve(Ytest, Yscore_test)
    roc_auc = auc(fpr, tpr)
    return(num_true, roc_auc, time_cost)
########################################################################################
########################################################################################
########################################################################################
def OFSA_numexp(gen_data, FSA_para, mb_size, exp_times):
    ####################
    OFSA_results = np.zeros((exp_times, 4))
    ####################
    for i in range(exp_times):
        ######################
        ### seed (classification)
        ######################
        np.random.seed(1991 + i)
        ######################
        num_true, roc_auc, t_feasel = OFSA_feasel(gen_data, FSA_para, mb_size)
        OFSA_results[i, 0] = 1991 + i
        OFSA_results[i, 1] = num_true
        OFSA_results[i, 2] = roc_auc
        OFSA_results[i, 3] = t_feasel
        #######################
    return(OFSA_results)
########################################################################################
###
###################
### p = 1000, k = 100, n = 10000, 30000, 100000, beta = 1 
###################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":1}    
FSA_par = {"eta":0.01, "mu":5, "lbd":0.01, "N_iter":200, "pretr_time":30}
OFSA1000_beta1 = OFSA_numexp(gen_data, FSA_par, 1000, 100)
DR_exp = np.sum(OFSA1000_beta1[:,1] == gen_data["k"])
PCD_exp = np.mean(OFSA1000_beta1[:,1] / gen_data["k"])
auc_exp = np.mean(OFSA1000_beta1[:,2])
time_exp = np.mean(OFSA1000_beta1[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":1}    
FSA_par = {"eta":0.01, "mu":5, "lbd":0.01, "N_iter":200, "pretr_time":30}
OFSA10000_beta1 = OFSA_numexp(gen_data, FSA_par, 1000, 100)
DR_exp = np.sum(OFSA10000_beta1[:,1] == gen_data["k"])
PCD_exp = np.mean(OFSA10000_beta1[:,1] / gen_data["k"])
auc_exp = np.mean(OFSA10000_beta1[:,2])
time_exp = np.mean(OFSA10000_beta1[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":30000, "p":1000, "k":100, "alpha":1, "beta_star":1}    
FSA_par = {"eta":0.01, "mu":5, "lbd":0.01, "N_iter":200, "pretr_time":30}
OFSA30000_beta1 = OFSA_numexp(gen_data, FSA_par, 1000, 100)
DR_exp = np.sum(OFSA30000_beta1[:,1] == gen_data["k"])
PCD_exp = np.mean(OFSA30000_beta1[:,1] / gen_data["k"])
auc_exp = np.mean(OFSA30000_beta1[:,2])
time_exp = np.mean(OFSA30000_beta1[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":100000, "p":1000, "k":100, "alpha":1, "beta_star":1}    
FSA_par = {"eta":0.01, "mu":5, "lbd":0.01, "N_iter":200, "pretr_time":30}
OFSA100000_beta1 = OFSA_numexp(gen_data, FSA_par, 1000, 100)
DR_exp = np.sum(OFSA100000_beta1[:,1] == gen_data["k"])
PCD_exp = np.mean(OFSA100000_beta1[:,1] / gen_data["k"])
auc_exp = np.mean(OFSA100000_beta1[:,2])
time_exp = np.mean(OFSA100000_beta1[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
###
###################
### p = 1000, k = 100, n = 10000, 30000, 100000, 300000, 1000000, beta = 0.01 
###################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":0.01}    
FSA_par = {"eta":0.01, "mu":5, "lbd":0.01, "N_iter":200, "pretr_time":30}
OFSA1000_beta001 = OFSA_numexp(gen_data, FSA_par, 1000, 100)
DR_exp = np.sum(OFSA1000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OFSA1000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OFSA1000_beta001[:,2])
time_exp = np.mean(OFSA1000_beta001[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":0.01}    
FSA_par = {"eta":0.01, "mu":5, "lbd":0.01, "N_iter":200, "pretr_time":30}
OFSA10000_beta001 = OFSA_numexp(gen_data, FSA_par, 1000, 100)
DR_exp = np.sum(OFSA10000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OFSA10000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OFSA10000_beta001[:,2])
time_exp = np.mean(OFSA10000_beta001[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":30000, "p":1000, "k":100, "alpha":1, "beta_star":0.01}    
FSA_par = {"eta":0.01, "mu":5, "lbd":0.01, "N_iter":200, "pretr_time":30}
OFSA30000_beta001 = OFSA_numexp(gen_data, FSA_par, 1000, 100)
DR_exp = np.sum(OFSA30000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OFSA30000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OFSA30000_beta001[:,2])
time_exp = np.mean(OFSA30000_beta001[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":100000, "p":1000, "k":100, "alpha":1, "beta_star":0.01}    
FSA_par = {"eta":0.01, "mu":5, "lbd":0.01, "N_iter":200, "pretr_time":30}
OFSA100000_beta001 = OFSA_numexp(gen_data, FSA_par, 1000, 100)
DR_exp = np.sum(OFSA100000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OFSA100000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OFSA100000_beta001[:,2])
time_exp = np.mean(OFSA100000_beta001[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":300000, "p":1000, "k":100, "alpha":1, "beta_star":0.01}    
FSA_par = {"eta":0.01, "mu":5, "lbd":0.01, "N_iter":200, "pretr_time":30}
OFSA300000_beta001 = OFSA_numexp(gen_data, FSA_par, 1000, 100)
DR_exp = np.sum(OFSA300000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OFSA300000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OFSA300000_beta001[:,2])
time_exp = np.mean(OFSA300000_beta001[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":1000000, "p":1000, "k":100, "alpha":1, "beta_star":0.01}    
FSA_par = {"eta":0.01, "mu":5, "lbd":0.01, "N_iter":200, "pretr_time":30}
OFSA1000000_beta001 = OFSA_numexp(gen_data, FSA_par, 1000, 100)
DR_exp = np.sum(OFSA1000000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OFSA1000000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OFSA1000000_beta001[:,2])
time_exp = np.mean(OFSA1000000_beta001[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###