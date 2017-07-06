import numpy as np
import scipy.optimize



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
def projection(moments):
    # project to a valid moment sequence on [-1,1]
    # assume the length of moments is odd
    n = len(moments)
    x = Variable(n)

    if n % 2 == 1:
        k = int((n+1)/2)
        A = Variable(k,k)
        B = Variable(k,k)
        constraints = [A+B>>0, A-B>>0]
        for i in range(k):
            for j in range(k):
                if i==0 and j==0:
                    constraints.append(A[0,0]==1)
                    constraints.append(B[0,0]==x[0])
                else:
                    constraints.append(A[i,j]==x[i+j-1])
                    constraints.append(B[i,j]==x[i+j])
    else:
        k = int(n/2)
        H = Variable(k,k)
        constraints = []
        for i in range(k):
            for j in range(k):
                if i==0 and j==0:
                    constraints.append(H[0,0]==1)
                else:
                    constraints.append(H[i,j]==x[i+j-1])
                    
    error = sum_squares(x - moments)
    objective = Minimize(error)
    prob = Problem(objective, constraints)
    prob.solve(solver=CVXOPT)
    
    return x.value.T.A1



from sympy import *
def mom_symbol(m):
    p, q= symbols('p q', nonnegative=True, real=True);
    x, y= symbols('x y', real=True);
    s = solve( [p+q-1, p*x+q*y-m[0], p*x*x+q*y*y-m[1], p*(x**3)+q*(y**3)-m[2]], [p,x,q,y])
    if (len(s)==0):
        print('No solution found!')
        print('Moments:', m)
        return False
    else:
        print('solution:',s[0])
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
