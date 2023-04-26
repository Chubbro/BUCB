# -*- coding: utf-8 -*-
import numpy as np

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
