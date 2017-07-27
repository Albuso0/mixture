import numpy as np
import scipy.optimize
import math
from discreteRV import finiteRV
import cvxpy
import sympy


def empiricalMoment(samples, degree):
    """ Compute empirical moments of samples
    input: samples x=(x1...xn), degree k
    output: empirical moments of x of degree up to k (start from first degree)
    """
    rawm = np.zeros(degree)
    monomial = np.ones(samples.shape)
    for i in range(degree):
        monomial *= samples
        rawm[i] = np.mean(monomial)

    return rawm


def HermiteMoment(m):
    """ Compute the moments of Hermite polynomials from raw moments
    input: moments of X of degree 0 to L
    output: moments of H_i(X) for i=0,...,L
    """
    m = np.asarray(m)
    l = len(m)
    HMom = np.zeros(m.shape)
    
    if ( l > 0 ):
        pp = np.zeros(m.shape); pp[0]=1
        HMom[0] = m[0]
    if ( l > 1 ):
        p = np.zeros(m.shape); p[1]=1
        HMom[1] = m[1]
    for k in range(2,l):
        # recursion: H_{n+1}(x) = x * H_n(x) - n * H_{n-1}(x)
        coeffs = np.roll(p, 1) - pp*(k-1)
        HMom[k] = np.dot(m, coeffs)
        pp = p; p = coeffs
        
    return HMom


def deconvolution(m):
    """ Deconvolve the moments of U+Z
    input: moments of U+Z of degree 1 to L
    output: moments of U of degree 1 to L
    """
    return HermiteMoment(np.insert(m,0,1))[1::]



    

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



def mom_symbol(m):
    """ Symbolic solver for ordinary method of moments 

    Args:
    m: a sequence of estimated moments, starting from degree 1
       only use odd number of moments

    Returns:
    U(finiteRV): estimated RV, whose moments is m
    
    """
    
    n = len(m)
    if n % 2 == 0:
        n = n-1 # only use 2k-1 moments
    k = int((n+1)/2)
    
    p = sympy.symbols('p0:%d'%(k), nonnegative=True, real=True);
    x = sympy.symbols('x0:%d'%(k), real=True);

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
    s = sympy.solve(equations,var)
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
        return finiteRV()
    else:
        return finiteRV(prob=s[0][0::2],val=s[0][1::2])

    
# def mom_numeric(m):
#     def equation(measure):
#         return(np.subtract(moment(measure, 3), m))
    
#     x0 = [0.5,3,0.5,1]
#     # x = scipy.optimize.fsolve(equation, x0)
#     x = scipy.optimize.broyden2(equation, x0)
#     print(x)
#     print(equation(x))


def quadmom(m):
    """ compute quadrature from moments
    ref: Gene Golub, John Welsch, Calculation of Gaussian Quadrature Rules

    Args:
    m: a valid moments sequence

    Returns:
    U: quadrature
    """

    m = np.asarray(m)
    ## TODO: check positive definite and decide to use how many moments
    if len(m) % 2 == 1:
        m = np.append(m,1e10)
    n = int(len(m)/2)
    m = np.insert(m,0,1)


    h = scipy.linalg.hankel(m[:n+1:], m[n::]) # Hankel matrix
    r = np.transpose(np.linalg.cholesky(h)) # upper triangular Cholesky factor

    # Compute alpha and beta from r, using Golub and Welsch's formula.
    alpha = np.zeros(n)
    alpha[0] = r[0][1] / r[0][0]
    for i in range(1,n):
        alpha[i] = r[i][i+1]/r[i][i] - r[i-1][i]/r[i-1][i-1]

    beta = np.zeros(n-1)
    for i in range(n-1):
        beta[i]=r[i+1][i+1]/r[i][i]

    jacobi = np.diag(alpha,0) + np.diag(beta,1) + np.diag(beta,-1)

    eigval, eigvec = np.linalg.eig(jacobi)
    
    x = eigval
    w = m[0] * np.power(eigvec[0],2)
    
    return finiteRV(prob=w, val=x)



def EM(samples, theta0, eps=1e-6, printIter=False, maxIter=5000):
    """ EM for estimating U under model U+Z

    Args:
    theta0 (finiteRV): initial estimate
    eps(float): termination accuracy
    printIter(bool): print results in each iteration 
    maxIter(int): maximum iterations

    Returns:
    theta(finiteRV): estimated model
    iterN(int): number of iterations

    """
    
    k = len(theta0.p)
    n = len(samples)

    curP = theta0.p
    curX = theta0.x
    samples = np.asarray(samples)

    iterN = 0
    while(True):
        preP = curP
        preX = curX

        T = np.zeros((n,k))
        for i in range(n):
            for j in range(k):
                T[i][j]= preP[j]*math.exp( -((samples[i]-preX[j])**2)/2 )
            T[i] /= sum(T[i])

        N = np.sum(T,axis=0)

        iterN += 1
        curP = N/np.sum(N)
        curX = np.divide( np.matmul(samples,T), N )

        if printIter:
            print(curP,curX)

        if ( iterN > maxIter or np.linalg.norm(np.subtract(curP,preP))+np.linalg.norm(np.subtract(curX,preX))<eps ):
            break

    return finiteRV(curP,curX), iterN
