# -*- coding: utf-8 -*-
#################################
## Running averages time
## Lizhe Sun
#################################
#################################
### Load Package
#################################
import numpy as np
import time
import onlineFSA
import eqcorrdata
#################################
#################################
#################################
def runningave_time_numexp(gen_dat, batch_size, loop_time):
    ##############
    ### Parameters
    ##############
    n = gen_dat["n"]
    p = gen_dat["p"]
    k = gen_dat["k"]
    alpha = gen_dat["alpha"]
    beta_star = gen_dat["beta_star"]
    result_mat = np.zeros((loop_time, 2))
    #############################
    num_batch = int(n / batch_size)
    ### Looping
    for i in range(loop_time):
        ###set seed
        np.random.seed(1991 + i)
        ## Store seed
        result_mat[i, 0] = i + 1991
        ########
        ### Generate data
        ### initial value for runningsums
        ########
        n_sum = 0
        Sx_sum = np.zeros((1, p))
        Sy_sum = 0
        Sxx_sum = np.zeros((p, p))
        Sxy_sum = np.zeros((p, 1))
        Syy_sum = 0
        rs_sum = {"n":n_sum, "Sx":Sx_sum, "Sy":Sy_sum, "Sxx":Sxx_sum, "Sxy":Sxy_sum, "Syy":Syy_sum}
        gen_data_partial = {"n":batch_size, "p":p, "k":k, "alpha":alpha, "beta_star":beta_star}
        ra_time = 0
        for j in range(num_batch):
            #############################
            X_batch, Y_batch, betastar_vec, istar = eqcorrdata.eqcorrdat_cls(gen_data_partial)
            #############################
            time_start = time.clock()
            rs_temp = onlineFSA.running_aves(X_batch, Y_batch)
            rs_sum = onlineFSA.add_runningaves(rs_sum, rs_temp)
            time_end = time.clock()
            cost_time = time_end - time_start
            #############################
            ra_time = ra_time + cost_time
        ####
        ####
        del X_batch, Y_batch
        ####
        if rs_sum["n"] != n:
            print("error")
            break
        #####
        result_mat[i, 1] = ra_time
        #####
    return(result_mat)
###########################################################################
###########################################################################
###########################################################################
#############
### beta = 1
#############
###    
########################################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":1}
result_10000_beta1 = runningave_time_numexp(gen_data, 1000, 100)
runningave_time_exp1 = np.mean(result_10000_beta1[:,1])
print(runningave_time_exp1)
###
########################################################################################
###
gen_data = {"n":30000, "p":1000, "k":100, "alpha":1, "beta_star":1}
result_30000_beta1 = runningave_time_numexp(gen_data, 1000, 100)
runningave_time_exp2 = np.mean(result_30000_beta1[:,1])
print(runningave_time_exp2)
###
########################################################################################
###
gen_data = {"n":100000, "p":1000, "k":100, "alpha":1, "beta_star":1}
result_100000_beta1 = runningave_time_numexp(gen_data, 1000, 100)
runningave_time_exp3 = np.mean(result_100000_beta1[:,1])
print(runningave_time_exp3)
###
########################################################################################
###
#############
### beta = 0.01
#############
###
########################################################################################
###
gen_data = {"n":10000, "p":1000, "k":100, "alpha":1, "beta_star":0.01}
result_10000_beta001 = runningave_time_numexp(gen_data, 1000, 100)
runningave_time_exp1 = np.mean(result_10000_beta001[:,1])
print(runningave_time_exp1)
###
########################################################################################
###
gen_data = {"n":30000, "p":1000, "k":100, "alpha":1, "beta_star":0.01}
result_30000_beta001 = runningave_time_numexp(gen_data, 1000, 100)
runningave_time_exp2 = np.mean(result_30000_beta001[:,1])
print(runningave_time_exp2)
###
########################################################################################
###
gen_data = {"n":100000, "p":1000, "k":100, "alpha":1, "beta_star":0.01}
result_100000_beta001 = runningave_time_numexp(gen_data, 1000, 100)
runningave_time_exp3 = np.mean(result_100000_beta001[:,1])
print(runningave_time_exp3)
###
########################################################################################
###
gen_data = {"n":300000, "p":1000, "k":100, "alpha":1, "beta_star":0.01}
result_300000_beta001 = runningave_time_numexp(gen_data, 1000, 100)
runningave_time_exp4 = np.mean(result_300000_beta001[:,1])
print(runningave_time_exp4)
###
########################################################################################
###
gen_data = {"n":1000000, "p":1000, "k":100, "alpha":1, "beta_star":0.01}
result_1000000_beta001 = runningave_time_numexp(gen_data, 1000, 100)
runningave_time_exp5 = np.mean(result_1000000_beta001[:,1])
print(runningave_time_exp5)
###
########################################################################################
###