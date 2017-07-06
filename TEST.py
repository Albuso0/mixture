import numpy as np
from functions import *



        


    

# m0=moment([ 0.5,  3,  0.5,   4],3)
# m1=moment([ 0.14741079,  4.58324761,  0.8483172,   3.32939126],3)
# m2=moment([ 0.14741079,  4.58324761,  1-0.14741079,   3.32939126],3)

# mom_symbol([1, 3.5, 12.5, 45.5])



# m=[0.0097461177746222157, -0.011894608372078452, -0.050834706264109228]
# print(m)
# print(projection(m))






m = moment([0.000205538730271, -6.11160027408564, 0.999794461269729, 0.00462288573198274],3)
print('moments:',m)
mom_symbol(m)
print('moments:',projection(m))
mom_symbol(projection(m,-1,1))


# print(HermiteMoments([1,1,2,4,10]))
# print(deconvolution([1,2,4,10]))



# from scipy.linalg import hankel
# m.insert(0,1)
# # print(m)
# h1 = hankel(m[0:2:1],m[1:3:1])
# h2 = hankel(m[1:3:1],m[2:4:1])
# print(np.linalg.eigvals(10*h1-h2))


succO = 0
succD = 0
Total = 0
for i in range(Total):
    x = np.random.randn(1, 10000)

    # raw moments: moments of X=U+Z
    rawm = []
    monomial = x
    for i in range(3):
        rawm.append(np.mean(monomial))
        monomial = np.multiply(monomial,x)
        
    # deconvolution: moments of U
    m = deconvolution(rawm)
    
    
    print('Ordinary MoM')
    if (mom_symbol(m)):
        succO = succO+1
    

    print('Denoised MoM')
    if(mom_symbol(projection(m))):
        succD = succD+1

    print('\n')

print('success in ordinary MoM: %d/%d'%(succO,Total))
print('success in denoised MoM: %d/%d'%(succD,Total))


