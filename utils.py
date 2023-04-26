# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import rv_continuous, beta, truncnorm


def trunc_norm(N, mu):
    sigma = np.full(N, 5)
    lower, upper = 1, 2
    dist_dic = {}
    for i in range(N): 
        X = truncnorm((lower - mu[i]) / sigma[i], (upper - mu[i]) / sigma[i], loc=mu[i], scale=sigma[i])
        dist_dic[i] = X        
    return dist_dic