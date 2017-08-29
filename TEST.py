import numpy as np
from solvers import *
from discreteRV import * 
from GM import *



# m = [0,2,0.00,15]
# m, sigma2 = estimate_sigma(m)
# print(m,sigma2)
# U = quadmom(m)
# print(U.p, U.x)


# k = 4


GM = modelGM(prob = [0.5,0.5], mean = [0,0], std = [1,np.sqrt(3)])
# GM = modelGM(prob = [0.5,0.5], mean = [0,1], std = 0.5)
x = sampleGM(GM, 1000000)
k = 2
m = empiricalMoment(x, 2*k)
m, sigma2 = estimate_sigma(m)
U = quadmom(m)
print('p= ', ' '.join(map(str,U.p)))
print('x= ', ' '.join(map(str,U.x)))
print('sigma= ', np.sqrt(sigma2))
print('sigma2= ', sigma2)

# rep = 20
# sample = (1+np.arange(10))*(500)

# print("Estimate: ")
# for n in sample:
#     print('n= ', n)
#     for i in range(rep):
#         x = sampleGM(GM, n)

#         #### EM
#         maxLL = float('-inf')
#         for rdCount in range(5):
#             start = finiteRV(prob=np.random.dirichlet(np.ones(k)), val=np.random.uniform(-1,1,k))
#             emRVcand,iterNcand = EM(x, start, tol=1e-3, printIter=False, maxIter=5000)
#             curLL = LL(x,emRVcand)
#             if curLL > maxLL:
#                 iterN = iterNcand
#                 emRV = emRVcand
#                 maxLL = curLL
#         print('p= ', ' '.join(map(str,emRV.p)))
#         print('x= ', ' '.join(map(str,emRV.x)))
#         print('iter: ', iterN)
        

        #### DMOM
        # m = deconvolution(empiricalMoment(x, 2*k-1))
        # proj = projection(m,-10,10)
        # dmomRV = quadmom(proj,dettol=0)
        # print('p= ', ' '.join(map(str,dmomRV.p)))
        # print('x= ', ' '.join(map(str,dmomRV.x)))
        



# mom_symbol([1, 3.5, 12.5, 45.5])


# U = finiteRV( [1], [0] )
# U = finiteRV( prob=[0.2, 0.5, 0.3], val=[3, 2, 1] )
# print(moment(U,4))
# V = finiteRV( prob=[0.2, 0.5, 0.3], val=[0, 2, 1.2] )
# print(W1(U,V))



# print(HermiteMoment([1,1,2,4,10,26,76,232]))
# print(deconvolution([1,2,4,10,26,76,232]))



# from scipy.linalg import hankel
# m.insert(0,1)
# # print(m)
# h1 = hankel(m[0:2:1],m[1:3:1])
# h2 = hankel(m[1:3:1],m[2:4:1])
# print(np.linalg.eigvals(10*h1-h2))


