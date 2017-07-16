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



    

# from cvxpy import *
import cvxpy
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

    
    x = cvxpy.Variable(n) # variables [m1,m2,...]
    obj = cvxpy.Minimize(cvxpy.sum_squares(x - moments)) # objective function

    # the following gives constraints
    # Ref for PSD condition: [Lasserre 2009, Theorem 3.3 and 3.4]
    if n % 2 == 1:
        # odd case
        k = int((n+1)/2)
        H = cvxpy.Variable(k,k+1)
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
        H = cvxpy.Variable(k,k)
        constraints = [H>>0, (a+b)*H[:k-1,1:]>>a*b*H[:k-1,:k-1]+H[1:,1:]]
        for i in range(k):
            for j in range(k):
                if i==0 and j==0:
                    constraints.append(H[0,0]==1)
                else:
                    constraints.append(H[i,j]==x[i+j-1])
                    
    
    prob = cvxpy.Problem(obj, constraints)
    opt = prob.solve(solver=cvxpy.CVXOPT,abstol=1e-10)
    
    return x.value.T.A1



from sympy import symbols, solve
def mom_symbol(m):
    # input: a sequence of estimated moments, starting from degree 1
    # output format: [p1,x1,p2,x2,...]
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
        ########### truncate a real number to rational
        # eq = -nsimplify(m[i], tolerance=0.1)
        # eq = float('%.2g' % -m[i])
        for j in range(k):
            eq = eq + p[j]*(x[j]**(i+1))
        equations.append(eq)

    var = [x for xs in zip(p, x) for x in xs] # format: [p1,x1,p2,x2,...]
    s = solve(equations,var)
    # s = solveset(equations,list(p)+list(x))
    # s = nonlinsolve(equations,list(p)+list(x))
    
    ########## some other tools in SymPy
    # print(equations)
    # s = solve_poly_system(equations,list(p)+list(x))
    # s = solve_triangulated(equations,list(p)+list(x))
    # s = solve_triangulated(equations,p[0],p[1],x[0],x[1])

    ########## test over simple rational coefficients
    # testEq = [x[0]*p[0] - 2*p[0], 2*p[0]**2 - x[0]**2]
    # s = solve_triangulated(testEq, x[0], p[0])
    
    
    if (len(s)==0):
        return []
    else:
        return [x for x in s[0]]

    
# def mom_numeric(m):
#     def equation(measure):
#         return(np.subtract(moment(measure, 3), m))
    
#     x0 = [0.5,3,0.5,1]
#     # x = scipy.optimize.fsolve(equation, x0)
#     x = scipy.optimize.broyden2(equation, x0)
#     print(x)
#     print(equation(x))



import subprocess
def quadrature(m):
    # return empty if the moment matrix is not PD (even if it is PSD)
    # TODO: resolve the case of PSD
    # output format: [p1,x1,p2,x2,...]
    args = ['./quadmom/quadmom']
    args.extend([str(x) for x in m])
    try:
        p = subprocess.check_output(args, universal_newlines=True) 
        res = [float(s) for s in p.split()] 
        return res
    except:
        res = []
        return res

    
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


def empiricalMoment(samples, degree):
    # input: samples x=(x1...xn), degree k
    # output: empirical moments of x of degree up to k (start from first degree)
    rawm = []
    monomial = samples
    for i in range(degree):
        rawm.append(np.mean(monomial))
        monomial = np.multiply(monomial,samples)
    return rawm


import math
def EM(samples, theta0, eps=1e-6, output=False):
    # input: theta0 = [p1,x1,p2,x2...], eps= termination accuracy
    # if output= True, will print results in each iteration 
    # output: theta = [p1,x1,p2,x2...]
    k = int( (len(theta0)+1)/2 )
    n = len(samples)

    curP = np.asarray(theta0[0::2])
    curX = np.asarray(theta0[1::2])
    samples = np.asarray(samples)
    
    while(True):
        preP = curP
        preX = curX

        T = np.zeros((n,k))
        for i in range(n):
            for j in range(k):
                T[i][j]= preP[j]*math.exp( -((samples[i]-preX[j])**2)/2 )
            T[i]=np.divide(T[i],sum(T[i]))

        N = np.sum(T,axis=0)

        curP = N/np.sum(N)
        curX = np.divide( np.matmul(samples,T), N )

        if output:
            print(curP,curX)

        if ( np.linalg.norm(np.subtract(curP,preP))+np.linalg.norm(np.subtract(curX,preX))<eps ):
            break


    return [x for xs in zip(curP, curX) for x in xs] # format: [p1,x1,p2,x2,...]
