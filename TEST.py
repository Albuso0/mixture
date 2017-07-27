import numpy as np
from solvers import *
from discreteRV import * 
from GM import *


# m0=moment([ 0.5,  3,  0.5,   4],3)
# m1=moment([ 0.14741079,  4.58324761,  0.8483172,   3.32939126],3)
# m2=moment([ 0.14741079,  4.58324761,  1-0.14741079,   3.32939126],3)

# mom_symbol([1, 3.5, 12.5, 45.5])



# m=[0.0097461177746222157, -0.011894608372078452, -0.050834706264109228]
# print(m)
# print(projection(m))




# print(HermiteMoment([1,1,2,4,10,26,76,232]))
# print(deconvolution([1,2,4,10,26,76,232]))


# U = finiteRV( [1], [0] )
# U = finiteRV( prob=[0.2, 0.5, 0.3], val=[0, 2, 1] )
# print(moment(U,4))
# V = finiteRV( prob=[0.2, 0.5, 0.3], val=[0, 2, 1.2] )
# print(W1(U,V))


n = 10000
k = 2
for expN in range(1):
    GM = modelGM( prob=[0.5, 0.5], mean=[-1, 10] )
    x = sampleGM(GM, n)

    # EM
    start = finiteRV(prob=[0.5,0.5], val=[-1,1])
    emRV,iterN = EM(x, start, eps=1e-6, printIter=False, maxIter=5000)
    print('EM:', emRV.p, emRV.x,  W1(emRV, GM.meanRV()) )
    
    # estimate moments of U
    m = deconvolution(empiricalMoment(x, 2*k-1))

    # ordinary MoM
    momRV = mom_symbol(m)
    print('Ordinary MoM:', momRV.p, momRV.x, W1(momRV, GM.meanRV()) )
    
    # Denoised MoM
    dmomRV = quadmom(projection(m,-2,12))
    print('Denoised MoM:', dmomRV.p, dmomRV.x, W1(dmomRV, GM.meanRV()) )
    
    print('\n')




# print('\n')

# k = 2;
# trueP = [0.000205538730271, -6.11160027408564, 0.999794461269729, 0.00462288573198274]; # format: [p1,x1,p2,x2...]
# m = moment(trueP, 2*k-1)
# mp = projection(m,-10,1)
# print('moments:',m)
# print('projected moments:',mp)
# print('Denoised MoM [p1,x1,p2,x2...]=',quadrature(mp))
# print('Ordinary MoM [p1,x1,p2,x2...]=', mom_symbol(m))






# print(HermiteMoments([1,1,2,4,10]))
# print(deconvolution([1,2,4,10]))



# from scipy.linalg import hankel
# m.insert(0,1)
# # print(m)
# h1 = hankel(m[0:2:1],m[1:3:1])
# h2 = hankel(m[1:3:1],m[2:4:1])
# print(np.linalg.eigvals(10*h1-h2))






############################################################
################### repeated experiments ###################
############################################################
Total = 0 # total number of experiments
k = 2

succO = 0
succD = 0
for i in range(Total):
    # samples
    x = np.random.randn(1, 10000)
    x = x[0]

    # raw moments: moments of X=U+Z of degree up to 2k-1
    rawm = empiricalMoment(x, 2*k-1)
    
    # deconvolution: moments of U
    m = deconvolution(rawm)
    

    # ordinary MoM
    pMOM = mom_symbol(m)
    print('Ordinary MoM:', pMOM)
    if (len(pMOM)>0):
        succO = succO+1
        
    # Denoised MoM
    pDMOM = quadrature(projection(m))
    print('Denoised MoM:', pDMOM)
    if (len(pDMOM)>0):
        succD = succD+1

    print('\n')
    
if Total>0:
    print('success in ordinary MoM: %d/%d'%(succO,Total))
    print('success in denoised MoM: %d/%d'%(succD,Total))


