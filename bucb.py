# -*- coding: utf-8 -*-
import numpy as np

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
