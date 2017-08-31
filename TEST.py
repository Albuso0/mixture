import numpy as np
from solvers import *
from discreteRV import * 
from GM import *
import time


n=10

samples = np.arange(n)/n
mu = np.array([-1,0])
sigma = np.array([1,2])
# precision = 1./(sigma**2) # inverse of sigma^2
# print(precision)
# LL0 = mu**2*precision - 2 * np.outer(samples, mu*precision) + np.outer(samples**2,precision)
# print(-0.5*(np.log(2*np.pi)+LL0)-np.log(sigma))




start_time = time.time()
GM = modelGM([0.5,0.5], mu, sigma)
print(LL2(samples,GM))
print("Time: %s seconds" % (time.time() - start_time))





start_time = time.time()
Total = 0.
for i in range(n):
    cumL = 0.
    for j in range(2):
        LL = (np.log(1/(np.sqrt(2*np.pi)*sigma[j]))-0.5*((samples[i]-mu[j])/sigma[j])**2)
        cumL += 0.5 * np.exp(LL)
    Total += np.log(cumL)
print(Total)
print("Time: %s seconds" % (time.time() - start_time))



print(EM2(samples,GM,printIter=True))
