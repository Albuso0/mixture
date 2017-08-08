import numpy as np
from solvers import *
from discreteRV import * 
from GM import *


import sys
import time

k = 5

# for i in range(20):
#     print("Model ",i+1)
#     print("p= ",' '.join( map(str,np.random.dirichlet(np.ones(k)))))
#     print("x= ",' '.join( map(str,np.random.uniform(-1,1,k))))
# exit()

with open('result2/Models.txt') as f:
    models = f.readlines()

model = int(sys.argv[1])
## fetch model
print("Model: ")
GM = modelGM(prob = np.array(models[(model-1)*3+1].split()[1:]).astype(np.float), mean = np.array(models[(model-1)*3+2].split()[1:]).astype(np.float))
# GM = modelGM( prob= np.random.dirichlet(np.ones(k)), mean=np.random.uniform(-1,1,k) )
print('p= ', ' '.join( map(str, GM.p)))
print('x= ', ' '.join( map(str, GM.mu)))
print('moments: ',' '.join( map(str,moment(GM.meanRV(),2*k-1))))



rep = 20
sample = (1+np.arange(10))*(500)

print("Estimate: ")
for n in sample:
    print('n= ', n)
    start_time = time.time()
    for i in range(rep):
        x = sampleGM(GM, n)

        #### EM
        # maxLL = float('-inf')
        # for rdCount in range(5):
        #     start = finiteRV(prob=np.random.dirichlet(np.ones(k)), val=np.random.uniform(-1,1,k))
        #     emRVcand,iterNcand = EM(x, start, tol=1e-3, printIter=False, maxIter=5000)
        #     curLL = LL(x,emRVcand)
        #     if curLL > maxLL:
        #         iterN = iterNcand
        #         emRV = emRVcand
        #         maxLL = curLL
        # print('p= ', ' '.join(map(str,emRV.p)))
        # print('x= ', ' '.join(map(str,emRV.x)))
        # print('iter: ', iterN)
        

        #### DMOM
        m = deconvolution(empiricalMoment(x, 2*k-1))
        proj = projection(m,-20,20)
        print('moments: ', ' '.join(map(str,m)))
        print('projected: ',' '.join(map(str,proj)))
        dmomRV = quadmom(proj,dettol=0)
        print('p= ', ' '.join(map(str,dmomRV.p)))
        print('x= ', ' '.join(map(str,dmomRV.x)))
        
    print("Time: %s seconds" % (time.time() - start_time))








        

n = 10000
k = 2

##### initial guess for iterative algorithms
# start = finiteRV(prob=1./k * np.ones(k), val=np.zeros(k))
start = finiteRV(prob=1./k * np.ones(k), val=np.arange(k)/(k-1)) 


for expN in range(0):
    GM = modelGM( prob=[0.6, 0.4], mean=[-1, 2] )
    # GM = modelGM( prob=[1], mean=[0.4] )
    x = sampleGM(GM, n)

    ###### EM
    # emRV,iterN = EM(x, start, eps=1e-9, printIter=True, maxIter=5000)
    # print('EM:', emRV.p, emRV.x,  W1(emRV, GM.meanRV()) )
    
    ###### estimate moments of U
    m = deconvolution(empiricalMoment(x, 2*k-1))

    ###### ordinary MoM - symbolic solver
    # momRV = mom_symbol(m,k)
    # print('Ordinary MoM:', momRV.p, momRV.x, W1(momRV, GM.meanRV()) )

    
    ###### ordinary MoM - numerical solver
    # momRV = mom_numeric(m, GM.meanRV()) # True value as initial guess
    # momRV = mom_numeric(m, start)  
    # print('Ordinary MoM:', momRV.p, momRV.x, W1(momRV, GM.meanRV()) )

    ###### EM as initial guess for ordinary MoM
    # emRV,iterN = EM(x, start, eps=1e-6, printIter=True, maxIter=30)
    # print('EM:', emRV.p, emRV.x,  W1(emRV, GM.meanRV()) )
    # momRV = mom_numeric(m, emRV)
    # print('Ordinary MoM:', momRV.p, momRV.x, W1(momRV, GM.meanRV()) )
    
    ###### Denoised MoM
    dmomRV = quadmom(projection(m,-2,12))
    print('Denoised MoM:', dmomRV.p, dmomRV.x, W1(dmomRV, GM.meanRV()) )
    
    print('\n')








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


