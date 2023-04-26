# -*- coding: utf-8 -*-
import numpy as np

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