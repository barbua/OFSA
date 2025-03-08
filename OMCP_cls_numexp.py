# -*- coding: utf-8 -*-
###################################
## Online MCP Feature selection numerical experiments
## Lizhe Sun
###################################
#################################
### Load Package
#################################
import numpy as np
import time
import sys
sys.path.append("/Users/lizhesun/Documents/OFSelection_2023/simulations/classification")
import math
import onlineFSA
import eqcorrdata
import OnlineRegularizedMethod
from sklearn.metrics import roc_curve, auc
#################################
##########
## Online MCP numerical experiment 
##########
def OMCP_feasel(gen_dat, OMCP_para, batch_size, lbd):
    ##########################
    n = gen_dat["n"]
    p = gen_dat["p"]
    k = gen_dat["k"]
    alpha = gen_dat["alpha"]
    beta_star = gen_dat["beta_star"]
    ###################
    ### lbd_para
    ###################
    eta = OMCP_para["eta"]
    b = OMCP_para["b"]
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
    ####
    lbd_start = math.exp(OMCP_para["start"])
    lbd_end = math.exp(OMCP_para["end"])
    ####
    ####################################################################################
    ####
    t_start = time.process_time()
    ####
    for Iter in range(100):
        lbd_mid = (lbd_start + lbd_end) / 2
        beta = OnlineRegularizedMethod.onlineMCP_cls(XX_normalize, XY_normalize, n, lbd_mid, eta, b, 500)
        beta_sel = np.flatnonzero(beta)
        sel_number = beta_sel.shape[0]
        ###################################
        if sel_number == k:
            break
        elif sel_number > k:
            lbd_start = lbd_mid
        else:
            lbd_end = lbd_mid
    ########
    beta_MCP = OnlineRegularizedMethod.onlineMCP_cls(XX_normalize, XY_normalize, n, lbd_mid, eta, b, 500)
    t_end = time.process_time()
    ########
    time_cost = t_end - t_start
    sel = np.flatnonzero(beta_MCP)
    print(sel.shape[0])
    num_true = len(np.intersect1d(istar, sel))
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
    beta_MCP = np.zeros((p, 1))
    beta_MCP[sel] = beta_sel
    ######
    ######
    Yscore_test = testX_standardize.dot(beta_MCP)
    fpr, tpr, thresholds = roc_curve(testY, Yscore_test)
    roc_auc = auc(fpr, tpr)
    ######
    return(num_true, roc_auc, time_cost)
########################################################################################
########################################################################################
########################################################################################
def OMCP_numexp(gen_data, OMCP_para, batch_size, lbd, exp_times):
    ####################
    OMCP_results = np.zeros((exp_times, 4))
    ####################
    for i in range(exp_times):
        ######################
        ### seed (classification)
        ######################
        np.random.seed(1991 + i)
        ######################
        num_true, roc_auc, t_feasel = OMCP_feasel(gen_data, OMCP_para, batch_size, lbd)
        OMCP_results[i, 0] = 1991 + i
        OMCP_results[i, 1] = num_true
        OMCP_results[i, 2] = roc_auc
        OMCP_results[i, 3] = t_feasel
        #######################
    return(OMCP_results)
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
OMCP_para = {"eta":0.001, "b":100, "start":-10, "end":10}    
OMCP1000_beta1 = OMCP_numexp(gen_data, OMCP_para, 1000, 0.01, 100)
DR_exp = np.sum(OMCP1000_beta1[:,1] == gen_data["k"])
PCD_exp = np.mean(OMCP1000_beta1[:,1] / gen_data["k"])
auc_exp = np.mean(OMCP1000_beta1[:,2])
time_exp = np.mean(OMCP1000_beta1[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":1}
OMCP_para = {"eta":0.001, "b":100, "start":-10, "end":10}    
OMCP10000_beta1 = OMCP_numexp(gen_data, OMCP_para, 1000, 0.01, 100)
DR_exp = np.sum(OMCP10000_beta1[:,1] == gen_data["k"])
PCD_exp = np.mean(OMCP10000_beta1[:,1] / gen_data["k"])
auc_exp = np.mean(OMCP10000_beta1[:,2])
time_exp = np.mean(OMCP10000_beta1[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":30000, "p":1000, "k":100, "alpha":1, "beta_star":1}
OMCP_para = {"eta":0.001, "b":100, "start":-10, "end":10}    
OMCP30000_beta1 = OMCP_numexp(gen_data, OMCP_para, 1000, 0.01, 100)
DR_exp = np.sum(OMCP30000_beta1[:,1] == gen_data["k"])
PCD_exp = np.mean(OMCP30000_beta1[:,1] / gen_data["k"])
auc_exp = np.mean(OMCP30000_beta1[:,2])
time_exp = np.mean(OMCP30000_beta1[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":100000, "p":1000, "k":100, "alpha":1, "beta_star":1}
OMCP_para = {"eta":0.001, "b":100, "start":-10, "end":10}    
OMCP100000_beta1 = OMCP_numexp(gen_data, OMCP_para, 1000, 0.01, 100)
DR_exp = np.sum(OMCP100000_beta1[:,1] == gen_data["k"])
PCD_exp = np.mean(OMCP100000_beta1[:,1] / gen_data["k"])
auc_exp = np.mean(OMCP100000_beta1[:,2])
time_exp = np.mean(OMCP100000_beta1[:,3] / gen_data["k"]) 
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
OMCP_para = {"eta":0.001, "b":100, "start":-10, "end":10}    
OMCP1000_beta001 = OMCP_numexp(gen_data, OMCP_para, 1000, 0.01, 100)
DR_exp = np.sum(OMCP1000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OMCP1000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OMCP1000_beta001[:,2])
time_exp = np.mean(OMCP1000_beta001[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":0.01}
OMCP_para = {"eta":0.001, "b":100, "start":-10, "end":10}    
OMCP10000_beta001 = OMCP_numexp(gen_data, OMCP_para, 1000, 0.01, 100)
DR_exp = np.sum(OMCP10000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OMCP10000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OMCP10000_beta001[:,2])
time_exp = np.mean(OMCP10000_beta001[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":30000, "p":1000, "k":100, "alpha":1, "beta_star":0.01}
OMCP_para = {"eta":0.001, "b":100, "start":-10, "end":10}    
OMCP30000_beta001 = OMCP_numexp(gen_data, OMCP_para, 1000, 0.01, 100)
DR_exp = np.sum(OMCP30000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OMCP30000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OMCP30000_beta001[:,2])
time_exp = np.mean(OMCP30000_beta001[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":100000, "p":1000, "k":100, "alpha":1, "beta_star":0.01}
OMCP_para = {"eta":0.001, "b":100, "start":-10, "end":10}    
OMCP100000_beta001 = OMCP_numexp(gen_data, OMCP_para, 1000, 0.01, 100)
DR_exp = np.sum(OMCP100000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OMCP100000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OMCP100000_beta001[:,2])
time_exp = np.mean(OMCP100000_beta001[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":300000, "p":1000, "k":100, "alpha":1, "beta_star":0.01}
OMCP_para = {"eta":0.001, "b":100, "start":-10, "end":10}    
OMCP300000_beta001 = OMCP_numexp(gen_data, OMCP_para, 1000, 0.01, 100)
DR_exp = np.sum(OMCP300000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OMCP300000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OMCP300000_beta001[:,2])
time_exp = np.mean(OMCP300000_beta001[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
###
########################################################################################
###
gen_data = {"n":1000000, "p":1000, "k":100, "alpha":1, "beta_star":0.01}
OMCP_para = {"eta":0.001, "b":100, "start":-10, "end":10}    
OMCP1000000_beta001 = OMCP_numexp(gen_data, OMCP_para, 1000, 0.01, 100)
DR_exp = np.sum(OMCP1000000_beta001[:,1] == gen_data["k"])
PCD_exp = np.mean(OMCP1000000_beta001[:,1] / gen_data["k"])
auc_exp = np.mean(OMCP1000000_beta001[:,2])
time_exp = np.mean(OMCP1000000_beta001[:,3] / gen_data["k"]) 
print("DR:", DR_exp)
print("PCD:", PCD_exp)
print("AUC:", auc_exp)
print("time", time_exp)
