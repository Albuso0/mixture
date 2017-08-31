import numpy as np
import scipy.optimize
import math
from discreteRV import finiteRV, moment
import cvxpy
import sympy
from scipy.linalg import hankel

def estimate_sigma(m):
    """ Lindsay's estimator, fit moments with U+sigma Z. Compute common sigma, and moments of U
    """
    
    if len(m) % 2 == 1:
        m = np.asarray(m[:-1])
    else:
        m = np.asarray(m)
    m = np.insert(m,0,1)

    l = len(m)
    HMom = [0]*l
    x = sympy.symbols('x', nonnegative=True, real=True); # x = sigma^2
    
    if ( l > 0 ):
        pp = np.zeros(m.shape); pp[0]=1
        HMom[0] = m[0]
    if ( l > 1 ):
        p = np.zeros(m.shape); p[1]=1
        HMom[1] = m[1]
    for k in range(2,l):
        # recursion: H_{n+1}(x) = x * H_n(x) - n * H_{n-1}(x)
        coeffs = np.roll(p, 1) - pp*(k-1)*x
        # print(coeffs)
        for i in range(k+1):
            HMom[k]+= coeffs[i]*m[i]
        # HMom[k] = np.dot(m, coeffs)
        pp = p; p = coeffs

    # Solve the first non-negative root
    # print(HMom)
    H = hankel(HMom[:int((l+1)/2)],HMom[int((l-1)/2):])
    eq = determinant(H).as_poly()
    # print(eq)
    
    coeffs = eq.coeffs()
    root = np.roots(coeffs)
    # root = sympy.solve(eq,x)
    root = root[np.isreal(root)].real
    root = root[root>=0]
    root.sort()
    # print(root)
    x0 = root[0]

    for k in range(2,l):
        HMom[k] = float(HMom[k].subs(x,x0))

    return (np.asarray(HMom[1:]), float(x0))
    

def determinant(M,r=[],c=[]):
    if len(r)==0:
        rows, cols = M.shape
        r = list(range(rows))
        c = list(range(cols))
        
    rows = len(r)
    cols = len(c)
    assert rows == cols

    if rows == 1:
        return M[r[0], c[0]]
    elif rows == 2:
        return M[r[0], c[0]]*M[r[1], c[1]] - M[r[0], c[1]]*M[r[1], c[0]]
    elif rows == 3:
        return (M[r[0], c[0]]*M[r[1], c[1]]*M[r[2], c[2]] + M[r[0], c[1]]*M[r[1], c[2]]*M[r[2], c[0]] + M[r[0], c[2]]*M[r[1], c[0]]*M[r[2], c[1]]) - \
               (M[r[0], c[2]]*M[r[1], c[1]]*M[r[2], c[0]] + M[r[0], c[0]]*M[r[1], c[2]]*M[r[2], c[1]] + M[r[0], c[1]]*M[r[1], c[0]]*M[r[2], c[2]])
    else:
        det = 0
        newr = r[1:]
        sign = 1
        for k in range(cols):
            newc = c[:k] + c[(k + 1):]
            det += determinant(M,newr,newc)*M[r[0],c[k]]*sign
            sign *= -1
        return det
    
    #### Available methods for computing determinant using Sympy: bareis, berkowitz, det_LU
    #### Method 1: berkowitz
    # eq = sympy.Matrix(H).det(method='berkowitz').as_poly().expand()
    #### Method 2: bareis
    # eq = sympy.Matrix(H).det(method='bareis').as_poly()
    # for i in eq.gens[1:]:
    #     eq = eq.eval(i,1)
    # return eq
    
        
        
    
    


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
    """ project to a valid moment sequence on [a,b]

    Args:
    a sequence of estimated moments, starting from degree 1

    Returns:
    a sequence of valid moments on [a,b], starting from degree 1 (default [a,b]=[-1,1])
    """
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
        constraints = [H[:,1:]-a*H[:,:k]>>0, b*H[:,:k]-H[:,1:]>>0]
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
        constraints = [H>>0, (a+b)*H[:k-1,1:]-a*b*H[:k-1,:k-1]+H[1:,1:]>>0]
        for i in range(k):
            for j in range(k):
                if i==0 and j==0:
                    constraints.append(H[0,0]==1)
                else:
                    constraints.append(H[i,j]==x[i+j-1])
                    
    
    prob = cvxpy.Problem(obj, constraints)
    opt = prob.solve(solver=cvxpy.CVXOPT)

    
    return x.value.T.A1



def mom_symbol(m, k):
    """ Symbolic solver for ordinary method of moments - find k-point representative distribution

    Args:
    m: a sequence of estimated moments, starting from degree 1
       only use first 2k-1 moments
    k: number of atoms in the output RV

    Returns:
    U(finiteRV): estimated RV, whose first 2k-1 moments equal m
    
    """
    
    n = 2*k-1
    assert len(m)>=n
    
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


    
def mom_numeric(m, start):
    """ numerical solver for ordinary method of moments - find k-point representative distribution

    Args:
    m: a sequence of estimated moments, starting from degree 1
       only use first 2k-1 moments
    start: initial guess. k = the number of components in start.

    Returns:
    U(finiteRV): estimated RV
    
    """
    k = len(start.p)
    n = 2*k-1
    assert len(m)>=n
    m = m[0:n]
    
    def equation(x):
        return moment(finiteRV(val=x[0:k],prob=np.append(x[k:],1-np.sum(x[k:]))),n)-m

    x0 = np.concatenate( (start.x, start.p[0:-1]) )
    x = scipy.optimize.fsolve(equation, x0)
    # x = scipy.optimize.broyden2(equation, x0)
    
    return finiteRV(val=x[0:k],prob=np.append(x[k:],1-np.sum(x[k:])))





def quadmom(m, dettol=0):
    """ compute quadrature from moments
    ref: Gene Golub, John Welsch, Calculation of Gaussian Quadrature Rules

    Args:
    m: a valid moments sequence

    Returns:
    U: quadrature
    """

    m = np.asarray(m)
    # INF = float('inf')
    INF = 1e10

    
    if len(m) % 2 == 1:
        m = np.append(m,INF)
    n = int(len(m)/2)
    m = np.insert(m,0,1)


    h = scipy.linalg.hankel(m[:n+1:], m[n::]) # Hankel matrix
    for i in range(len(h)):
        # check positive definite and decide to use how many moments
        if np.linalg.det(h[0:i+1,0:i+1])<=dettol: # alternative: less than some threshold
            h = h[0:i+1,0:i+1]
            h[i,i] = INF
            n = i
            break
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




def LLmatrix(samples, means):
    """
    Log-likelihood matrix of samples under the given means
    
    Args:
    samples: array
    means: array

    Return:
    LL: log-likelihood, shape(len(samples), len(means))
    """

    samples = np.asarray(samples)
    means = np.asarray(means)
    
    LL0 = means**2 - 2 * np.outer(samples, means) + (samples**2)[:, np.newaxis]
    return -0.5*(np.log(2*np.pi)+LL0)


def LL(samples, U):
    """
    Log-likelihood matrix of samples under the given means
    
    Args:
    samples: array
    means: array

    Return:
    LL: log-likelihood, shape(len(samples), len(means))
    """
    return np.sum( np.log( np.dot(np.exp(LLmatrix(samples,U.x)), U.p) ))



def EM(samples, theta0, tol=1e-3, printIter=False, maxIter=100):
    """ EM for estimating U under model U+Z

    Args:
    theta0 (finiteRV): initial estimate
    tol(float): termination accuracy
    printIter(bool): print results in each iteration 
    maxIter(int): maximum iterations

    Returns:
    theta(finiteRV): estimated model
    iterN(int): number of iterations

    """
    
    k = len(theta0.p)
    n = len(samples)
    samples = np.asarray(samples)

    iterN = 0
    curP = theta0.p
    curX = theta0.x
    LLmat = LLmatrix(samples, curX)
    curLL = np.sum( np.log( np.dot(np.exp(LLmat), curP) ))

    while(True):
        preLL = curLL
        
        T = np.exp(LLmat) * curP
        T /= np.sum(T,axis=1)[:, np.newaxis]

        N = np.sum(T,axis=0)

        iterN += 1
        curP = N/np.sum(N)
        curX = np.divide( np.matmul(samples,T), N )
        LLmat = LLmatrix(samples, curX)
        curLL = np.sum( np.log( np.dot(np.exp(LLmat), curP) ))

        if printIter:
            print(curP,curX)
            print(curLL)

        if ( iterN > maxIter or curLL-preLL<tol ):
            break

    return finiteRV(curP,curX), iterN




def LLmatrix2(samples, model):
    """
    Log-likelihood matrix of samples under the given GM model
    
    Args:
    samples: array
    model: a GM instance

    Return:
    LL: log-likelihood, shape(len(samples), number of components)
    """

    samples = np.asarray(samples)
    
    precision = 1./(model.sigma**2) # inverse of sigma^2
    LL0 = model.mu**2*precision - 2 * np.outer(samples, model.mu*precision) + np.outer(samples**2,precision)
    return (-0.5*(np.log(2*np.pi)+LL0)-np.log(model.sigma))
    

def LL2(samples, model):
    """
    Log-likelihood matrix of samples under the given GM model
    
    Args:
    samples: array
    model: a GM instance

    Return:
    LL: log-likelihood
    """
    return np.sum( np.log( np.dot(np.exp(LLmatrix2(samples,model)), model.p) ))


def EM2(samples, model0, tol=1e-3, printIter=False, maxIter=100):
    """ EM for estimating U under model U+Z

    Args:
    model0 (modelGM): initial guess
    tol(float): termination accuracy
    printIter(bool): print results in each iteration 
    maxIter(int): maximum iterations

    Returns:
    model(modelGM): estimated model
    iterN(int): number of iterations

    """
    
    k = len(model0.p)
    n = len(samples)
    samples = np.asarray(samples)

    iterN = 0
    model = model0
    Lmat = np.exp(LLmatrix2(samples, model))
    curLL = np.sum( np.log( np.dot(Lmat, model.p) ))

    while(True):
        preLL = curLL
        
        T = Lmat * model.p
        T /= np.sum(T,axis=1)[:, np.newaxis]

        N = np.sum(T,axis=0)

        iterN += 1
        model.p = N/np.sum(N)
        model.mu = np.divide( np.matmul(samples,T), N )
        
        cross = model.mu**2 - 2 * np.outer(samples, model.mu) + (samples**2)[:, np.newaxis]
        sigma2 = np.sum(cross*T)/n
        model.sigma = np.ones(model.p.shape) * np.sqrt(sigma2)
        
        Lmat = np.exp(LLmatrix2(samples, model))
        curLL = np.sum( np.log( np.dot(Lmat, model.p) ))

        if printIter:
            print(model.p,model.mu,model.sigma)
            print(curLL)

        if ( iterN > maxIter or curLL-preLL<tol ):
            break

    return model, iterN, curLL

