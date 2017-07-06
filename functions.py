import numpy as np
import scipy.optimize
import math


def HermiteMoments(m):
    # input: moments of X of degree 0 to L
    # output: moments of H_i(X) for i=0,...,L
    l = len(m)
    ans = [0]*l
    if ( l > 0 ):
        # pp = np.zeros(l)
        pp = [0]*l
        pp[0] = 1
        ans[0] = m[0]
    if ( l > 1 ):
        p = [0]*l
        p[1] = 1
        ans[1] = m[1]
    for k in range(2,l):
        # recursion: H_{n+1}(x) = x * H_n(x) - n * H_{n-1}(x)
        coeffs = np.roll(p, 1) - np.multiply(pp,k-1)
        ans[k] = np.dot(m, coeffs)
        pp = p
        p = coeffs
    return ans


def deconvolution(m):
    # input: moments of U+Z of degree 1 to L
    # output: moments of U of degree 1 to L
    m.insert(0,1)
    mout = HermiteMoments(m)
    return mout[1::]



    

from cvxpy import *
def projection(moments, a=-1, b=1):
    # input: a sequence of estimated moments, starting from degree 1
    # output: a sequence of valid moments on [a,b], starting from degree 1 (default [a,b]=[-1,1])
    # function: project to a valid moment sequence on [a,b]
    n = len(moments)
    if n == 0:
        return moments
    if n == 1:
        moments[0] = max(a,min(b,moments[0]))
        return moments

    
    x = Variable(n) # variables [m1,m2,...]
    obj = Minimize(sum_squares(x - moments)) # objective function

    # the following gives constraints
    # Ref for PSD condition: [Lasserre 2009, Theorem 3.3 and 3.4]
    if n % 2 == 1:
        # odd case
        k = int((n+1)/2)
        H = Variable(k,k+1)
        constraints = [H[:,1:]>>a*H[:,:k], b*H[:,:k]>>H[:,1:]]
        for i in range(k):
            for j in range(k+1):
                if i==0 and j==0:
                    constraints.append(H[0,0]==1)
                else:
                    constraints.append(H[i,j]==x[i+j-1])
    else:
        # even case
        k = int(n/2)+1
        H = Variable(k,k)
        constraints = [H>>0, (a+b)*H[:k-1,1:]>>a*b*H[:k-1,:k-1]+H[1:,1:]]
        for i in range(k):
            for j in range(k):
                if i==0 and j==0:
                    constraints.append(H[0,0]==1)
                else:
                    constraints.append(H[i,j]==x[i+j-1])
                    
    
    prob = Problem(obj, constraints)
    opt = prob.solve(solver=CVXOPT,abstol=1e-10)
    
    return x.value.T.A1



from sympy import * 
def mom_symbol(m):
    n = len(m)
    if n % 2 == 0:
        n = n-1 # only use 2k-1 moments
    k = int((n+1)/2)
    
    p = symbols('p0:%d'%(k), nonnegative=True, real=True);
    x = symbols('x0:%d'%(k), real=True);

    eq = -1
    for i in range(k):
        eq = eq + p[i]
    equations = [eq]

    for i in range(n):
        eq = -m[i]
        for j in range(k):
            eq = eq + p[j]*(x[j]**(i+1))
        equations.append(eq)
    s = solve(equations,list(p)+list(x))
    # s = nonlinsolve(equations,list(p)+list(x))
    
    if (len(s)==0):
        print('No solution found!')
        print('Moments:', m)
        return False
    else:
        print('solution:',s[0])  # format: (p1,p2,...,x1,x2,...)
        return True



    
# def mom_numeric(m):
#     def equation(measure):
#         return(np.subtract(moment(measure, 3), m))
    
#     x0 = [0.5,3,0.5,1]
#     # x = scipy.optimize.fsolve(equation, x0)
#     x = scipy.optimize.broyden2(equation, x0)
#     print(x)
#     print(equation(x))
    


    
def moment(distribution, degree):
    # Return the moments of the input distribution up to the given degree (start from first degree)
    # distribution: [p1,x1,p2,x2,...]
    mass = distribution[0::2]
    atom = distribution[1::2]
    ans = []

    monomial = atom
    for n in range(degree):
        ans.append(np.dot(mass, monomial))
        monomial = np.multiply(monomial,atom)
    return ans
