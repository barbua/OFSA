# -*- coding: utf-8 -*-
###################################
## Lasso feature selection
## Lizhe Sun
###################################
###################################
### Load Package
###################################
import numpy as np
import math
from sklearn.linear_model import Lasso
####
## Lasso parameter
#### 
def Lasso_feature_sel(trainX, trainY, lbd_para, k):
    ## Set alpha_list
    lbd_start = math.exp(lbd_para["start"])
    lbd_end = math.exp(lbd_para["end"])
    ###
    for iter in range(100):
        lbd_mid = (lbd_start + lbd_end) / 2
        ### Because trainX and trainY will be centered, the intercept is removed
        Lasso_model = Lasso(alpha = lbd_mid, fit_intercept = False)
        fit_model = Lasso_model.fit(trainX, trainY)
        temp_beta = fit_model.coef_
        temp_beta_sel = np.flatnonzero(temp_beta)
        sel_number = temp_beta_sel.shape[0]
        if sel_number == k:
            break
        elif sel_number > k:
            lbd_start = lbd_mid
        else:
            lbd_end = lbd_mid
    ##  
    #############################################################################
    ##
    Lasso_model = Lasso(alpha = lbd_mid, fit_intercept = False)
    fit_model = Lasso_model.fit(trainX, trainY)
    beta_lasso = fit_model.coef_
    beta_sel_index = np.flatnonzero(beta_lasso)
    print(beta_sel_index.shape[0])
    #####
    #####
    return(beta_sel_index, lbd_mid)
          
            