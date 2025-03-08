# -*- coding: utf-8 -*-
######################################################
### OPG classification numerical experiment
######################################################
#############
### import package
#############
import numpy as np
import sys
sys.path.append("/Users/lizhesun/Documents/OFSelection_2023/simulations/classification")
import eqcorrdata
import OPG_cls
import time
from sklearn.metrics import roc_curve, auc
#############
########################################################################################
########################################################################################
def OPG_feasel(gen_data, lbd_par, batch_size, eta, lbd_l2, mb_size):
    #################
    #################
    n = gen_data["n"]
    p = gen_data["p"]
    k = gen_data["k"]
    alpha = gen_data["alpha"]
    beta_star = gen_data["beta_star"]
    #################
    #################
    num_batch = int(n / batch_size)
    ###########
    ###lbd
    ###########
    lbd_start = lbd_par["lbd_start"]
    lbd_end = lbd_par["lbd_end"]
    lbd_vec = np.exp(np.linspace(lbd_start, lbd_end, 200))
    ###########
    beta_mat = np.zeros((p, lbd_vec.shape[0]))
    beta0_vec = np.zeros(lbd_vec.shape[0])
    sel_fea_number = np.zeros(lbd_vec.shape[0])
    #################################
    time_cost = 0
    #################################
    #################################
    gen_data_batch = {"n":batch_size, "p":p, "k":k, "alpha":alpha, "beta_star":beta_star}
    #################################
    for i in range(num_batch):
        ############
        ### Generate Data
        ############
        Xtr_batch, Ytr_batch, betastar_vec, istar = eqcorrdata.eqcorrdat_cls(gen_data_batch)
        ############
        time_start = time.process_time()
        for Iter in range(lbd_vec.shape[0]):
            ################################
            lbd = lbd_vec[Iter]
            beta = beta_mat[:, Iter].reshape(p, 1)
            beta0 = beta0_vec[Iter]
            ################################
            beta, beta0 = OPG_cls.OPG_Lasso(Xtr_batch, Ytr_batch, beta, beta0, lbd, lbd_l2, eta, mb_size)
            ################################
            sel_fea_number[Iter] = np.flatnonzero(beta).shape[0]
            beta_mat[:, Iter] = beta.ravel()
            beta0_vec[Iter] = beta0
            ################################
        time_end = time.process_time()
        t_cost = time_end - time_start
        time_cost = time_cost + t_cost
    ########################################
    lbd_index = np.where(sel_fea_number <= k)[0]
    lbd_sel_index = lbd_index[0]
    print(sel_fea_number[lbd_sel_index])
    beta_OPG = beta_mat[:, lbd_sel_index].reshape(p, 1)
    beta0_OPG = beta0_vec[lbd_sel_index]
    beta_index = np.flatnonzero(beta_OPG)
    num_true = len(np.intersect1d(istar, beta_index))
    #############
    ### Test data    
    #############
    testdat_para = {"n":batch_size, "p":p, "k":k, "alpha":alpha, "beta_star":beta_star}
    Xtest, Ytest, betastar_vec, istar = eqcorrdata.eqcorrdat_cls(testdat_para)
    Yscore_test = Xtest.dot(beta_OPG) + beta0_OPG * np.ones((batch_size, 1))
    fpr, tpr, thresholds = roc_curve(Ytest, Yscore_test)
    roc_auc = auc(fpr, tpr)
    ##########
    ##########
    return(num_true, roc_auc, time_cost)
########################################################################################
########################################################################################
########################################################################################
def OPG_cls_numexp(gen_data, lbd_par, batch_size, eta, lbd_l2, mb_size, exp_times):
    ####################
    OPG_results = np.zeros((exp_times, 4))
    ####################
    for i in range(exp_times):
        ######################
        ### seed (classification)
        ######################
        np.random.seed(1991 + i)
        ######################
        num_true, roc_auc, t_feasel = OPG_feasel(gen_data, lbd_par, batch_size, eta, lbd_l2, mb_size)
        ######################
        OPG_results[i, 0] = 1991 + i
        OPG_results[i, 1] = num_true
        OPG_results[i, 2] = roc_auc
        OPG_results[i, 3] = t_feasel
        #######################
    return(OPG_results)
########################################################################################
###
###################
### p = 1000, k = 100, n = 10000, 30000, 100000, beta = 1 
###################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":1}
lbd_para = {"lbd_start":-2, "lbd_end":0}    
OPG1000_beta1 = OPG_cls_numexp(gen_data, lbd_para, 1000, 0.01, 0.01, 25, 100)
DR_exp = np.sum(OPG1000_beta1[:,1] == gen_data["k"])
PCD_exp = np.mean(OPG1000_beta1[:,1] / gen_data["k"])
auc_exp = np.mean(OPG1000_beta1[:,2])
time_exp = np.mean(OPG1000_beta1[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":1}
lbd_para = {"lbd_start":-2, "lbd_end":0}    
OPG10000_beta1 = OPG_cls_numexp(gen_data, lbd_para, 1000, 0.01, 0.01, 25, 100)
DR_exp = np.sum(OPG10000_beta1[:,1] == gen_data["k"])
PCD_exp = np.mean(OPG10000_beta1[:,1] / gen_data["k"])
auc_exp = np.mean(OPG10000_beta1[:,2])
time_exp = np.mean(OPG10000_beta1[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":30000, "p":1000, "k":100, "alpha":1, "beta_star":1}
lbd_para = {"lbd_start":-2, "lbd_end":0}    
OPG30000_beta1 = OPG_cls_numexp(gen_data, lbd_para, 1000, 0.01, 0.01, 25, 100)
DR_exp = np.sum(OPG30000_beta1[:,1] == gen_data["k"])
PCD_exp = np.mean(OPG30000_beta1[:,1] / gen_data["k"])
auc_exp = np.mean(OPG30000_beta1[:,2])
time_exp = np.mean(OPG30000_beta1[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":100000, "p":1000, "k":100, "alpha":1, "beta_star":1}
lbd_para = {"lbd_start":-2, "lbd_end":0}    
OPG100000_beta1 = OPG_cls_numexp(gen_data, lbd_para, 1000, 0.01, 0.01, 25, 100)
DR_exp = np.sum(OPG100000_beta1[:,1] == gen_data["k"])
PCD_exp = np.mean(OPG100000_beta1[:,1] / gen_data["k"])
auc_exp = np.mean(OPG100000_beta1[:,2])
time_exp = np.mean(OPG100000_beta1[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
###################
### p = 1000, k = 100, n = 10000, 30000, 100000, beta = 0.01 
###################
###
gen_data = {"n":1000, "p":1000, "k":100, "alpha":1, "beta_star":0.01}
lbd_para = {"lbd_start":-2, "lbd_end":0}    
OPG1000_beta001 = OPG_cls_numexp(gen_data, lbd_para, 1000, 0.01, 0.01, 25, 100)
DR_exp = np.sum(OPG1000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OPG1000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OPG1000_beta001[:,2])
time_exp = np.mean(OPG1000_beta001[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":0.01}
lbd_para = {"lbd_start":-2, "lbd_end":0}    
OPG10000_beta001 = OPG_cls_numexp(gen_data, lbd_para, 1000, 0.01, 0.01, 25, 100)
DR_exp = np.sum(OPG10000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OPG10000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OPG10000_beta001[:,2])
time_exp = np.mean(OPG10000_beta001[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":30000, "p":1000, "k":100, "alpha":1, "beta_star":0.01}
lbd_para = {"lbd_start":-2, "lbd_end":0}    
OPG30000_beta001 = OPG_cls_numexp(gen_data, lbd_para, 1000, 0.01, 0.01, 25, 100)
DR_exp = np.sum(OPG30000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OPG30000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OPG30000_beta001[:,2])
time_exp = np.mean(OPG30000_beta001[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":100000, "p":1000, "k":100, "alpha":1, "beta_star":1}
lbd_para = {"lbd_start":-2, "lbd_end":0}    
OPG100000_beta001 = OPG_cls_numexp(gen_data, lbd_para, 1000, 0.01, 0.01, 25, 100)
DR_exp = np.sum(OPG100000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OPG100000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OPG100000_beta001[:,2])
time_exp = np.mean(OPG100000_beta001[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":300000, "p":1000, "k":100, "alpha":1, "beta_star":1}
lbd_para = {"lbd_start":-2, "lbd_end":0}    
OPG300000_beta001 = OPG_cls_numexp(gen_data, lbd_para, 1000, 0.01, 0.01, 25, 100)
DR_exp = np.sum(OPG300000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OPG300000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OPG300000_beta001[:,2])
time_exp = np.mean(OPG300000_beta001[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":1000000, "p":1000, "k":100, "alpha":1, "beta_star":1}
lbd_para = {"lbd_start":-2, "lbd_end":0}    
OPG1000000_beta001 = OPG_cls_numexp(gen_data, lbd_para, 1000, 0.01, 0.01, 25, 100)
DR_exp = np.sum(OPG1000000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OPG1000000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OPG1000000_beta001[:,2])
time_exp = np.mean(OPG1000000_beta001[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
