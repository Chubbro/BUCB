import pandas as pd
import numpy as np
import random
import math
from scipy.stats import rv_continuous, beta, truncnorm
#from functools import wraps


def trunc_norm(N, mu):
    sigma = np.full(N, 5)
    lower, upper = 1, 2
    dist_dic = {}
    for i in range(N): 
        X = truncnorm((lower - mu[i]) / sigma[i], (upper - mu[i]) / sigma[i], loc=mu[i], scale=sigma[i])
        dist_dic[i] = X        
    return dist_dic

# Budgeted UCB
def BUCB(N, M, mu, c, B, reward, expense, t, T, u_hat, dist_dic):
    reward_next = reward
    T_next = T
    t_next = t
    u_hat_next = u_hat
    
    cmin = min(c)
    Bt = B - expense
    while Bt >= M*cmin:
        u_bar = u_hat + np.sqrt(2*np.log(t)/(T*M)) if t != 0 else 0
        
        if t < N:
            if Bt >= M*c[t]:
                I = t
            else:
                break
        else:
            I = (np.log(M*u_bar*(c*M <= Bt))/c).argmax()

        samples = dist_dic[I].rvs(M)
        
        if Bt >= M*c[(np.log(M*u_bar)/c).argmax()]:
            reward_next += np.log(sum(samples))
            u_hat_next[I] = (T[I]*M*u_hat[I]+sum(samples))/(T[I]*M+M)
            expense += M*c[I]
            T_next[I] += 1
            t_next += 1
            
        reward += np.log(sum(samples))     
        Bt = Bt - M*c[I]
        u_hat[I] = (T[I]*M*u_hat[I]+sum(samples))/(T[I]*M+M)
        T[I] += 1
        t += 1
              
    return u_hat_next, reward, reward_next, expense, t_next, T_next

# Oracle
def oracle(N, M, mu, c, B, reward, expense, t, T, u_hat, dist_dic):
    reward_next = reward
    T_next = T
    t_next = t
    u_hat_next = u_hat
    
    cmin = min(c)
    ratio = np.log(mu)/c
    I = ratio.argmax()
    n = int((B-expense)//(M*c[I]))
    
    for k in range(n):
        samples = dist_dic[I].rvs(M)
        reward += np.log(sum(samples))
        reward_next += np.log(sum(samples))
    
    t += n
    t_next += n
    T[I] += n
    T_next[I] += n
    Bt = B-expense-M*c[I]*n
    expense += M*c[I]*n
        
    while Bt>=M*cmin:
        J = (np.log(M*mu*(c*M <= Bt))/c).argmax()
        Bt = Bt - M*c[J]
        samples = dist_dic[J].rvs(M)
        reward += np.log(sum(samples))
        #expense += M*c[J]
        T[J] += 1
        t += 1
        
    return u_hat_next, reward, reward_next, expense, t_next, T_next

# Epsilon-greedy
def epsilon_greedy(rate):    # Closer
    def inner(N, M, mu, c, B, reward, expense, t, T, u_hat, dist_dic):
        cmin = min(c)
        Bt = B - expense

        while Bt >= M*cmin:        
            if t < N:
                if Bt >= M*c[t]:
                    I = t
                else:
                    break
            else:
                if np.random.choice([True, False], p=[rate,1-rate]):
                    I = np.random.choice(range(N))
                else:
                    I = (np.log(M*u_hat)/c).argmax()
        
            if Bt < M*c[I]:
                break

            samples = dist_dic[I].rvs(M)
            Bt = Bt - M*c[I]
            reward += np.log(sum(samples))
            expense += M*c[I]
            u_hat[I] = (T[I]*M*u_hat[I]+sum(samples))/(T[I]*M+M)
            T[I] += 1
            t += 1
            
        reward_next = reward
        T_next = T
        t_next = t
        u_hat_next = u_hat
        
        return u_hat_next, reward, reward_next, expense, t_next, T_next
    return inner    # Return a decorated function

# Adaptive epsilon_greedy (epsilon = 1/t)
def epx(N, M, mu, c, B, reward, expense, t, T, u_hat, dist_dic):
    cmin = min(c)
    Bt = B - expense

    while Bt >= M*cmin:
        rate = 1/t    # only difference in epx, adaptive manner.
        if t < N:
            if Bt >= M*c[t]:
                I = t
            else:
                break
        else:
            if np.random.choice([True, False], p=[rate,1-rate]): 
                I = np.random.choice(range(N))
            else:
                I = (np.log(M*u_hat)/c).argmax()    #I = (np.log(M*u_hat*(c*M <= Bt))/c).argmax()
        
        if Bt < M*c[I]:
            break
        
        
        samples = dist_dic[I].rvs(M)  # dist_dic[I].rvs(size=10) gives list of u_ij(t) for all j.
        Bt = Bt - M*c[I]
        reward += np.log(sum(samples))
        expense += M*c[I]
        u_hat[I] = (T[I]*M*u_hat[I]+sum(samples))/(T[I]*M+M)
        T[I] += 1
        t += 1
       
    reward_next = reward
    T_next = T
    t_next = t
    u_hat_next = u_hat

    return u_hat_next, reward, reward_next, expense, t_next, T_next

# Explore-then-commit
def ETC(rate):    # Closer
    def inner(N, M, mu, c, B, reward, expense, t, T, u_hat, dist_dic):
        cmin = min(c)
        Bt = B - expense
        # exploration
        while Bt >= M*cmin:           
            if expense < rate*B:
                I = t%N
                samples = dist_dic[I].rvs(M)
                u_hat[I] = (T[I]*M*u_hat[I]+sum(samples))/(T[I]*M+M)
            else:
                I = (np.log(M*u_hat)/c).argmax()
                samples = dist_dic[I].rvs(M)
            
            if Bt < M*c[I]:
                break
                    
            Bt = Bt - M*c[I]
            reward += np.log(sum(samples))
            expense += M*c[I]
            T[I] += 1
            t += 1
        
        reward_next = reward
        T_next = T
        t_next = t
        u_hat_next = u_hat
            
        return u_hat_next, reward, reward_next, expense, t_next, T_next
    return inner