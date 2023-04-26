# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random
from bucb import BUCB
from oracle import oracle
from epgreedy import epsilon_greedy, epx
from etc import ETC
from utils import trunc_norm



def get_reward(N, M, c_low, c_high, seed, func_tup):
    func_num = len(func_tup)
    func_name = [func_tup[x].__name__ for x in range(func_num)]
    para_name = ["random_seed", "c_range", "c", "mu", "N", "M", "B"]
    np.random.seed(seed)
    mu = np.random.uniform(low=1, high=2, size=N)
    c = np.random.uniform(low=c_low, high=c_high, size=N)
    cmin = min(c)
    para_value = [seed, [c_low,c_high], c, mu, N, M] 
    
    data = pd.DataFrame(columns=para_name+func_name)
    
    dist_dic = trunc_norm(N, mu)
    mu_bar = np.zeros(N)
    for k in range(N):
        mu_bar[k] = np.average(dist_dic[k].rvs(1000000))
    
    np.random.seed() 
    
    for s in range(10):
        t_all = np.zeros(func_num, dtype=int)
        reward_all = np.zeros(func_num)
        reward_record = np.zeros(func_num)
        expense_all = np.zeros(func_num)
        T_all = np.zeros((func_num, N), dtype=int)
        u_hat_all = np.zeros((func_num, N))
        u_bar = np.zeros(N)
        
        for B in range(0, 4000, 500):
            for i, func in zip(range(func_num), func_tup):
                u_hat_next, reward, reward_next, expense, t_next, T_next = func(N, M, mu_bar, c, B, reward_all[i], expense_all[i], t_all[i], T_all[i], u_hat_all[i], dist_dic)
                # update parameters
                u_hat_all[i] = u_hat_next
                reward_all[i] = reward_next
                reward_record[i] = reward
                expense_all[i] = expense
                t_all[i] = t_next
                T_all[i] = T_next
            
            new_row = pd.Series(np.concatenate([para_value, [B], reward_record]), index=para_name+func_name)
            data = data.append(new_row, ignore_index=True)
        
    return data


if __name__ == '__main__':
    ep1 = epsilon_greedy(rate=0.01)
    ep1.__name__ = "ep1"
    ep5 = epsilon_greedy(rate=0.05)
    ep5.__name__ = "ep5"
    ep10 = epsilon_greedy(rate=0.1)
    ep10.__name__ = "ep10"
    ep15 = epsilon_greedy(rate=0.15)
    ep15.__name__ = "ep15"
    etc5 = ETC(rate=0.05)
    etc5.__name__ = "etc5"
    etc10 = ETC(rate=0.10)
    etc10.__name__ = "etc10"
    etc15 = ETC(rate=0.15)
    etc15.__name__ = "etc15"
    etc20 = ETC(rate=0.20)
    etc20.__name__ = "etc20"
    etc25 = ETC(rate=0.25)
    etc25.__name__ = "etc25"
    etc30 = ETC(rate=0.30)
    etc30.__name__ = "etc30"
    
    d = get_reward(10, 5, 1, 2, 7, (oracle,BUCB,ep1,ep5,ep10,ep15,epx,etc15,etc20,etc25,etc30))