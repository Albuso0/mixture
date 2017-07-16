import numpy as np
from functions import *



        


    

# m0=moment([ 0.5,  3,  0.5,   4],3)
# m1=moment([ 0.14741079,  4.58324761,  0.8483172,   3.32939126],3)
# m2=moment([ 0.14741079,  4.58324761,  1-0.14741079,   3.32939126],3)

# mom_symbol([1, 3.5, 12.5, 45.5])



# m=[0.0097461177746222157, -0.011894608372078452, -0.050834706264109228]
# print(m)
# print(projection(m))






n = 10000
k = 2

############ X = p N(1,1) + (1-p) N(1,1)
# p = 0.8
# z = np.random.binomial(n, p)
# x = np.random.randn(1, n)
# x = x[0]
# x[0:z:] += 1
# x = np.random.permutation(x)


############ X=Z
x = np.random.randn(1, n)
x = x[0]



start = [0.5,-1,0.5,1] # format:[p1,x1,p2,x2....]
pEM = EM(x, start, 1e-6, True)
weights = pEM[0::2]
atoms = pEM[1::2]
print('EM:', weights, atoms)



# raw moments: moments of X=U+Z of degree up to 2k-1
rawm = empiricalMoment(x, 2*k-1)

# deconvolution: moments of U
m = deconvolution(rawm)

# ordinary MoM
pMOM = mom_symbol(m)
weights = pMOM[0::2]
atoms = pMOM[1::2]
print('Ordinary MoM:', weights, atoms)
        
# Denoised MoM
pDMOM = quadrature(projection(m,-1,1))
weights = pDMOM[0::2]
atoms = pDMOM[1::2]
print('Denoised MoM:', weights, atoms)










print('\n')

k = 2;
trueP = [0.000205538730271, -6.11160027408564, 0.999794461269729, 0.00462288573198274]; # format: [p1,x1,p2,x2...]
m = moment(trueP, 2*k-1)
mp = projection(m,-10,1)
print('moments:',m)
print('projected moments:',mp)
print('Denoised MoM [p1,x1,p2,x2...]=',quadrature(mp))
print('Ordinary MoM [p1,x1,p2,x2...]=', mom_symbol(m))






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


