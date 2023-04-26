# -*- coding: utf-8 -*-
import numpy as np

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