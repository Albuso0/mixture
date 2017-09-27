"""
For tests
"""
import time
import numpy as np
from solvers import mom_symbol, empiricalMoment, deconvolution 
# from discreteRV import finiteRV
from GM import modelGM, sampleGM


def main_mom_bad():
    """
    script to show that mom fails
    """
    cnt = 0
    for i in range(100):
        np.random.seed(i)
        comp = 2
        gm_model = modelGM(prob=[0.5, 0.5], mean=[0.1, -0.1], std=1)
        num = 10
        sample = sampleGM(gm_model, num)
        m_raw = empiricalMoment(sample, 2*comp-1)
        m_dec = deconvolution(m_raw)
        mean = mom_symbol(m_dec, comp)
        print('Instance ', i)
        print('Samples=', sample)
        print('Raw moments= ', m_raw)
        print('Deconvolved moments= ', m_dec)
        print('Estimated means= ', mean.x)
        print('Estimated weights= ', mean.p)
        print('')
        if len(mean.x)==0:
            cnt +=1
    print(cnt)



if __name__ == '__main__':
    main_mom_bad()




def main_lindsay():
    """
    test Lindsay
    """
    k = 3
    GM = modelGM(prob = [1.], mean = [0.], std = 1)


    rep = 20
    sample = (1+np.arange(10))*(500)


    print("Estimate: ")
    for n in sample:
        print('n= ', n)
        start_time = time.time()
        for i in range(rep):
            np.random.seed(i)
            x = sampleGM(GM, n)
            
            #### Lindsay
            m = empiricalMoment(x, 2*k)
            # print('Sample moments= ', m)
            m, sigma2 = estimate_sigma(m)
            U = quadmom(m)
            print('p= ', ' '.join(map(str,U.p)))
            print('x= ', ' '.join(map(str,U.x)))
            print('sigma= ', np.sqrt(sigma2))


def main_other():
    """
    some other tests
    """
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
