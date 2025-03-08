# -*- coding: utf-8 -*-
#################################
### Yangzi Guo
### Truncated Online Feature Selection with Sparse Projection 
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
def OSTSGD_feasel(gen_data, OSTSGD_para, batch_size, mb_size):
    ####
    n = gen_data["n"]
    p = gen_data["p"]
    k = gen_data["k"]
    alpha = gen_data["alpha"]
    beta_star = gen_data["beta_star"]
    dat_type = gen_data["dat_type"]
    ####
    lbd = OSTSGD_para["lbd"]
    ############################
    ### initial value
    ############################
    ####
    beta = np.zeros((p + 1, 1))    ##### parameter vector
    sigma = np.ones((p + 1, 1))    ##### diagonal elements of covarince matrix
    total_time = 0
    ############################
    #### Feature selection and training procedure
    ############################
    num_batch = int(n/batch_size)
    gen_data_batch = {"n":batch_size, "p":p, "k":k, "alpha":alpha, "beta_star":beta_star, "dat_type":dat_type}
    for i in range(num_batch):
        ###
        ### Generate data
        ###
        Xtr_mb, Ytr_mb, betastar_vec, istar = eqcorrdata.eqcorrdat_cls(gen_data_batch)
        Xtr_mb = np.concatenate((Xtr_mb, np.ones((Xtr_mb.shape[0], 1))), axis = 1)
        ################
        start_time = time.process_time()
        beta, sel = gradhingeloss.OSTSGD_cls(Xtr_mb, Ytr_mb, beta, k, lbd, sigma, mb_size)
        end_time = time.process_time()
        time_cost = end_time - start_time
        total_time = total_time + time_cost
    ##########
    ##########
    num_true = len(np.intersect1d(istar, sel))
    ##########
    ##########
    ### Test data    
    ##########
    testdat_para = {"n":batch_size, "p":p, "k":k, "alpha":alpha, "beta_star":beta_star, "dat_type":dat_type}
    Xtest, Ytest, betastar_vec, istar = eqcorrdata.eqcorrdat_cls(testdat_para)
    Xtest = np.concatenate((Xtest, np.ones((Xtest.shape[0], 1))), axis=1)
    Yscore_test = Xtest.dot(beta)
    fpr, tpr, thresholds = roc_curve(Ytest, Yscore_test)
    roc_auc = auc(fpr, tpr)
    ##########
    return(num_true, roc_auc, total_time)
########################################################################################
########################################################################################
########################################################################################
def OSTSGD_numexp(gen_data, OSTSGD_para, batch_size, mb_size, exp_times):
    ####################
    OSTSGD_results = np.zeros((exp_times, 4))
    ####################
    for i in range(exp_times):
        ######################
        ### seed
        ######################
        np.random.seed(1991 + i) 
        ######################
        num_true, roc_auc, t_feasel = OSTSGD_feasel(gen_data, OSTSGD_para, batch_size, mb_size)
        OSTSGD_results[i, 0] = 1991 + i
        OSTSGD_results[i, 1] = num_true
        OSTSGD_results[i, 2] = roc_auc
        OSTSGD_results[i, 3] = t_feasel
        #######################
    return(OSTSGD_results)     
########################################################################################
###
###################
### p = 1000, k = 100, n = 10000, 30000, 100000, beta = 1 
###################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}  
OSTSGD_para = {"eta":0.01, "lbd":0.01}
OSTSGD_1000_beta1 = OSTSGD_numexp(gen_data, OSTSGD_para, 1000, 25, 100)
DR_exp = np.sum(OSTSGD_1000_beta1[:,1] == gen_data["k"])
PCD_exp = np.mean(OSTSGD_1000_beta1[:,1] / gen_data["k"])
auc_exp = np.mean(OSTSGD_1000_beta1[:,2])
time_exp = np.mean(OSTSGD_1000_beta1[:,3] / gen_data["k"]) 
print("########################################################################################")
print("########################################################################################")
print("Second Order Truncated Online Feature Selection#########################################")
print(gen_data)
print(OSTSGD_para)
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
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":1, "dat_type":1}  
OSTSGD_para = {"eta":0.01, "lbd":0.01}
OSTSGD_10000_beta1 = OSTSGD_numexp(gen_data, OSTSGD_para, 1000, 25, 100)
DR_exp = np.sum(OSTSGD_10000_beta1[:,1] == gen_data["k"])
PCD_exp = np.mean(OSTSGD_10000_beta1[:,1] / gen_data["k"])
auc_exp = np.mean(OSTSGD_10000_beta1[:,2])
time_exp = np.mean(OSTSGD_10000_beta1[:,3] / gen_data["k"]) 
print("########################################################################################")
print("########################################################################################")
print("Second Order Truncated Online Feature Selection#########################################")
print(gen_data)
print(OSTSGD_para)
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
OSTSGD_para = {"eta":0.01, "lbd":0.01}
OSTSGD_30000_beta1 = OSTSGD_numexp(gen_data, OSTSGD_para, 1000, 25, 100)
DR_exp = np.sum(OSTSGD_30000_beta1[:,1] == gen_data["k"])
PCD_exp = np.mean(OSTSGD_30000_beta1[:,1] / gen_data["k"])
auc_exp = np.mean(OSTSGD_30000_beta1[:,2])
time_exp = np.mean(OSTSGD_30000_beta1[:,3] / gen_data["k"]) 
print("########################################################################################")
print("########################################################################################")
print("Second Order Truncated Online Feature Selection#########################################")      
print(gen_data)
print(OSTSGD_para)
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
OSTSGD_para = {"eta":0.01, "lbd":0.01}
OSTSGD_100000_beta1 = OSTSGD_numexp(gen_data, OSTSGD_para, 1000, 25, 100)
DR_exp = np.sum(OSTSGD_100000_beta1[:,1] == gen_data["k"])
PCD_exp = np.mean(OSTSGD_100000_beta1[:,1] / gen_data["k"])
auc_exp = np.mean(OSTSGD_100000_beta1[:,2])
time_exp = np.mean(OSTSGD_100000_beta1[:,3] / gen_data["k"]) 
print("########################################################################################")
print("########################################################################################")
print("Second Order Truncated Online Feature Selection#########################################")
print(gen_data)
print(OSTSGD_para)
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
OSTSGD_para = {"eta":0.01, "lbd":0.01}
OSTSGD_1000_beta001 = OSTSGD_numexp(gen_data, OSTSGD_para, 1000, 25, 100)
DR_exp = np.sum(OSTSGD_1000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OSTSGD_1000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OSTSGD_1000_beta001[:,2])
time_exp = np.mean(OSTSGD_1000_beta001[:,3] / gen_data["k"]) 
print("########################################################################################")
print("########################################################################################")
print("Second Order Truncated Online Feature Selection#########################################")
print(gen_data)
print(OSTSGD_para)
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
OSTSGD_para = {"eta":0.01, "lbd":0.01}
OSTSGD_10000_beta001 = OSTSGD_numexp(gen_data, OSTSGD_para, 1000, 25, 100)
DR_exp = np.sum(OSTSGD_10000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OSTSGD_10000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OSTSGD_10000_beta001[:,2])
time_exp = np.mean(OSTSGD_10000_beta001[:,3] / gen_data["k"]) 
print("########################################################################################")
print("########################################################################################")
print("Second Order Truncated Online Feature Selection#########################################")
print(gen_data)
print(OSTSGD_para)
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
OSTSGD_para = {"eta":0.01, "lbd":0.01}
OSTSGD_30000_beta001 = OSTSGD_numexp(gen_data, OSTSGD_para, 1000, 25, 100)
DR_exp = np.sum(OSTSGD_30000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OSTSGD_30000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OSTSGD_30000_beta001[:,2])
time_exp = np.mean(OSTSGD_30000_beta001[:,3] / gen_data["k"]) 
print("########################################################################################")
print("########################################################################################")
print("Second Order Truncated Online Feature Selection#########################################")
print(gen_data)
print(OSTSGD_para)
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
OSTSGD_para = {"eta":0.01, "lbd":0.01}
OSTSGD_100000_beta001 = OSTSGD_numexp(gen_data, OSTSGD_para, 1000, 25, 100)
DR_exp = np.sum(OSTSGD_100000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OSTSGD_100000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OSTSGD_100000_beta001[:,2])
time_exp = np.mean(OSTSGD_100000_beta001[:,3] / gen_data["k"]) 
print("########################################################################################")
print("########################################################################################")
print("Second Order Truncated Online Feature Selection#########################################")
print(gen_data)
print(OSTSGD_para)
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
OSTSGD_para = {"eta":0.01, "lbd":0.01}
OSTSGD_300000_beta001 = OSTSGD_numexp(gen_data, OSTSGD_para, 1000, 25, 100)
DR_exp = np.sum(OSTSGD_300000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OSTSGD_300000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OSTSGD_300000_beta001[:,2])
time_exp = np.mean(OSTSGD_300000_beta001[:,3] / gen_data["k"]) 
print("########################################################################################")
print("########################################################################################")
print("Second Order Truncated Online Feature Selection#########################################")
print(gen_data)
print(OSTSGD_para)
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
OSTSGD_para = {"eta":0.01, "lbd":0.01}
OSTSGD_1000000_beta001 = OSTSGD_numexp(gen_data, OSTSGD_para, 1000, 25, 100)
DR_exp = np.sum(OSTSGD_1000000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OSTSGD_1000000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OSTSGD_1000000_beta001[:,2])
time_exp = np.mean(OSTSGD_1000000_beta001[:,3] / gen_data["k"]) 
print("########################################################################################")
print("########################################################################################")
print("Second Order Truncated Online Feature Selection#########################################")
print(gen_data)
print(OSTSGD_para)
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    