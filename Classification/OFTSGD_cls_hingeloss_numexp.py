# -*- coding: utf-8 -*-
#################################
### Yangzi Guo
### Truncated Online Feature Selection with Sparse Projection
### 
#################################
################
### Load Package
################
import numpy as np
import sys
sys.path.append("/Users/lizhesun/Documents/OFSelection_2023/simulations/classification")
import gradhingeloss
import eqcorrdata
import time
from sklearn.metrics import roc_curve, auc
#################################
### Numerical Experiment
#################################
def OFTSGD_feasel(gen_data, FTSGD_para, batch_size, mb_size):
    ####
    n = gen_data["n"]
    p = gen_data["p"]
    k = gen_data["k"]
    alpha = gen_data["alpha"]
    beta_star = gen_data["beta_star"]
    dat_type = gen_data["dat_type"]
    ####
    eta = FTSGD_para["eta"]
    lbd = FTSGD_para["lbd"]
    ############################
    ### initial value
    ############################
    ####
    beta = np.zeros((p, 1))  ##### parameter vector
    beta0 = 0 ##### intercept
    total_time = 0
    ############################
    #### Feature selection and training procedure
    ############################
    num_batch = int(n / batch_size)
    gen_data_batch = {"n":batch_size, "p":p, "k":k, "alpha":alpha, "beta_star":beta_star, "dat_type":dat_type}
    ####################
    for i in range(num_batch):
        ################
        ### Generate data
        ################
        Xtr_mb, Ytr_mb, betastar_vec, istar = eqcorrdata.eqcorrdat_cls(gen_data_batch)
        ################
        ### Update beta
        ################
        start_time = time.process_time()
        beta, beta0, sel = gradhingeloss.OFTSGD_cls(Xtr_mb, Ytr_mb, beta, beta0, k, lbd, eta, mb_size)
        end_time = time.process_time()
        time_cost = end_time - start_time
        total_time = total_time + time_cost
        ################
    ###########
    ##########
    num_true = len(np.intersect1d(istar, sel))
    ##########
    ##########
    ### Test data    
    ##########
    testdat_para = {"n":1000, "p":p, "k":k, "alpha":alpha, "beta_star":beta_star, "dat_type":dat_type}
    Xtest, Ytest, betastar_vec, istar = eqcorrdata.eqcorrdat_cls(testdat_para)
    Yscore_test = beta0 * np.ones((1000, 1)) + Xtest.dot(beta)
    fpr, tpr, thresholds = roc_curve(Ytest, Yscore_test)
    roc_auc = auc(fpr, tpr)
    ##########
    return(num_true, roc_auc, total_time)
########################################################################################
########################################################################################
########################################################################################
def OFTSGD_numexp(gen_data, FTSGD_para, batch_size, mb_size, exp_times):
    ####################
    FTSGD_results = np.zeros((exp_times, 4))
    ####################
    for i in range(exp_times):
        ######################
        ### seed
        ######################
        np.random.seed(1991 + i)
        ######################
        num_true, roc_auc, t_feasel = OFTSGD_feasel(gen_data, FTSGD_para, batch_size, mb_size)
        FTSGD_results[i, 0] = 1991 + i
        FTSGD_results[i, 1] = num_true
        FTSGD_results[i, 2] = roc_auc
        FTSGD_results[i, 3] = t_feasel
        ##########
        ### Second Step
        ##########
        #######################
    return(FTSGD_results)
###     
########################################################################################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}  
FTSGD_para = {"eta":0.01, "lbd":0.01}
FTSGD_1000_beta1 = OFTSGD_numexp(gen_data, FTSGD_para, 1000, 25, 100)
DR_exp = np.sum(FTSGD_1000_beta1[:,1] == gen_data["k"])
PCD_exp = np.mean(FTSGD_1000_beta1[:,1] / gen_data["k"])
auc_exp = np.mean(FTSGD_1000_beta1[:,2])
time_exp = np.mean(FTSGD_1000_beta1[:,3] / gen_data["k"]) 
print("########################################################################################")
print("########################################################################################")
print("First Order Truncated Online Feature Selection##########################################")
print(gen_data)
print(FTSGD_para)
print()
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
print()
print("########################################################################################")
print("########################################################################################")
###
###################
### p = 1000, k = 100, n = 10000, 30000, 100000, beta = 1 
###################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}  
FTSGD_para = {"eta":0.01, "lbd":0.01}
FTSGD_10000_beta1 = OFTSGD_numexp(gen_data, FTSGD_para, 1000, 25, 100)
DR_exp = np.sum(FTSGD_10000_beta1[:,1] == gen_data["k"])
PCD_exp = np.mean(FTSGD_10000_beta1[:,1] / gen_data["k"])
auc_exp = np.mean(FTSGD_10000_beta1[:,2])
time_exp = np.mean(FTSGD_10000_beta1[:,3] / gen_data["k"]) 
print("########################################################################################")
print("########################################################################################")
print("First Order Truncated Online Feature Selection##########################################")
print(gen_data)
print(FTSGD_para)
print()
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
print()
print("########################################################################################")
print("########################################################################################")
###
########################################################################################
###
gen_data = {"n":30000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}  
FTSGD_para = {"eta":0.01, "lbd":0.01}
FTSGD_30000_beta1 = OFTSGD_numexp(gen_data, FTSGD_para, 1000, 25, 100)
DR_exp = np.sum(FTSGD_30000_beta1[:,1] == gen_data["k"])
PCD_exp = np.mean(FTSGD_30000_beta1[:,1] / gen_data["k"])
auc_exp = np.mean(FTSGD_30000_beta1[:,2])
time_exp = np.mean(FTSGD_30000_beta1[:,3] / gen_data["k"]) 
print("########################################################################################")
print("########################################################################################")
print("First Order Truncated Online Feature Selection##########################################")
print(gen_data)
print(FTSGD_para)
print()
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
print()
print("########################################################################################")
print("########################################################################################")
###
########################################################################################
###
gen_data = {"n":100000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}  
FTSGD_para = {"eta":0.01, "lbd":0.01}
FTSGD_100000_beta1 = OFTSGD_numexp(gen_data, FTSGD_para, 1000, 25, 100)
DR_exp = np.sum(FTSGD_100000_beta1[:,1] == gen_data["k"])
PCD_exp = np.mean(FTSGD_100000_beta1[:,1] / gen_data["k"])
auc_exp = np.mean(FTSGD_100000_beta1[:,2])
time_exp = np.mean(FTSGD_100000_beta1[:,3] / gen_data["k"]) 
print("########################################################################################")
print("########################################################################################")
print("First Order Truncated Online Feature Selection##########################################")
print(gen_data)
print(FTSGD_para)
print()
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
print()
print("########################################################################################")
print("########################################################################################")
###
########################################################################################
###    
###
###################
### p = 1000, k = 100, n = 10000, 30000, 100000, 300000, 1000000, beta = 0.01 
###################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}  
FTSGD_para = {"eta":0.01, "lbd":0.01}
FTSGD_1000_beta001 = OFTSGD_numexp(gen_data, FTSGD_para, 1000, 25, 100)
DR_exp = np.sum(FTSGD_1000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(FTSGD_1000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(FTSGD_1000_beta001[:,2])
time_exp = np.mean(FTSGD_1000_beta001[:,3] / gen_data["k"]) 
print("########################################################################################")
print("########################################################################################")
print("First Order Truncated Online Feature Selection##########################################")
print(gen_data)
print(FTSGD_para)
print()
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
print()
print("########################################################################################")
print("########################################################################################")
###
########################################################################################
### 
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}  
FTSGD_para = {"eta":0.01, "lbd":0.01}
FTSGD_10000_beta001 = OFTSGD_numexp(gen_data, FTSGD_para, 1000, 25, 100)
DR_exp = np.sum(FTSGD_10000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(FTSGD_10000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(FTSGD_10000_beta001[:,2])
time_exp = np.mean(FTSGD_10000_beta001[:,3] / gen_data["k"]) 
print("########################################################################################")
print("########################################################################################")
print("First Order Truncated Online Feature Selection##########################################")
print(gen_data)
print(FTSGD_para)
print()
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
print()
print("########################################################################################")
print("########################################################################################")
###
########################################################################################
###        
gen_data = {"n":30000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}  
FTSGD_para = {"eta":0.01, "lbd":0.01}
FTSGD_30000_beta001 = OFTSGD_numexp(gen_data, FTSGD_para, 1000, 25, 100)
DR_exp = np.sum(FTSGD_30000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(FTSGD_30000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(FTSGD_30000_beta001[:,2])
time_exp = np.mean(FTSGD_30000_beta001[:,3] / gen_data["k"]) 
print("########################################################################################")
print("########################################################################################")
print("First Order Truncated Online Feature Selection##########################################")
print(gen_data)
print(FTSGD_para)
print()
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
print()
print("########################################################################################")
print("########################################################################################")
###
########################################################################################
###        
gen_data = {"n":100000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}  
FTSGD_para = {"eta":0.01, "lbd":0.01}
FTSGD_100000_beta001 = OFTSGD_numexp(gen_data, FTSGD_para, 1000, 25, 100)
DR_exp = np.sum(FTSGD_100000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(FTSGD_100000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(FTSGD_100000_beta001[:,2])
time_exp = np.mean(FTSGD_100000_beta001[:,3] / gen_data["k"]) 
print("########################################################################################")
print("########################################################################################")
print("First Order Truncated Online Feature Selection##########################################")
print(gen_data)
print(FTSGD_para)
print()
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
print()
print("########################################################################################")
print("########################################################################################")
###
########################################################################################
###        
gen_data = {"n":300000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}  
FTSGD_para = {"eta":0.01, "lbd":0.01}
FTSGD_300000_beta001 = OFTSGD_numexp(gen_data, FTSGD_para, 1000, 25, 100)
DR_exp = np.sum(FTSGD_300000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(FTSGD_300000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(FTSGD_300000_beta001[:,2])
time_exp = np.mean(FTSGD_300000_beta001[:,3] / gen_data["k"]) 
print("########################################################################################")
print("########################################################################################")
print("First Order Truncated Online Feature Selection##########################################")
print(gen_data)
print(FTSGD_para)
print()
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
print()
print("########################################################################################")
print("########################################################################################")
###
########################################################################################
###        
gen_data = {"n":1000000, "p":1000, "k":100, "alpha":1, "beta_star":0.01, "dat_type":1}  
FTSGD_para = {"eta":0.01, "lbd":0.01}
FTSGD_1000000_beta001 = OFTSGD_numexp(gen_data, FTSGD_para, 1000, 25, 100)
DR_exp = np.sum(FTSGD_1000000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(FTSGD_1000000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(FTSGD_1000000_beta001[:,2])
time_exp = np.mean(FTSGD_1000000_beta001[:,3] / gen_data["k"]) 
print("########################################################################################")
print("########################################################################################")
print("First Order Truncated Online Feature Selection##########################################")
print(gen_data)
print(FTSGD_para)
print()
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
print()
print("########################################################################################")
print("########################################################################################")
###
########################################################################################
###        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    