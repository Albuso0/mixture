"""
module for operations on moments
"""

import numpy as np
import cvxpy
from scipy.linalg import hankel
from discrete_rv import DiscreteRV

def empirical_moment(samples, degree):
    """ Compute empirical moments of samples
    Args:
    samples: x=(x1...xn)
    degree: L

    Returns
    matrix M of size k*n
    each row is the moments of (x1...xn) (start from the first degree to degree L)
    """
    m_raw = np.empty((degree, len(samples)))
    m_raw[0, :] = samples
    for i in range(1, degree):
        m_raw[i, :] = m_raw[i-1, :] * samples

    return m_raw

def hermite_transform_matrix(degree, sigma=1):
    """ Hermite transformation
    Let x=(x,...,x^k)', then Ax+b=(g_1(x,sigma),...,g_k(x,sigma))
    g_k(x,sigma)=sigma^k*H_k(x/sigma), g_k(x,1)=H_k(x), the usual Hermite polynomial

    Args:
    degree: int
    highest degree k

    Return:
    tuple (A,b): A is a matrix of shape (k,k), b is a vector of shape (k,1)
    """
    length = degree+1
    var = sigma*sigma
    mat = np.zeros((length, length))
    if length > 0:
        prepre = np.zeros(length)
        prepre[0] = 1
        mat[0, :] = prepre
    if length > 1:
        pre = np.zeros(length)
        pre[1] = 1
        mat[1, :] = pre
    for k in range(2, length):
        # recursion: H_{n+1}(x) = x * H_n(x) - n * H_{n-1}(x)
        # => g_{n+1}(x,s) = x * g_n(x,s) - n * s^2 * g_{n-1}(x,s)
        coeffs = np.roll(pre, 1) - prepre*(k-1)*var
        mat[k, :] = coeffs
        prepre = pre
        pre = coeffs

    return (mat[1:, 1:], mat[1:, 0].reshape((degree, 1)))



def projection(moments, int_a=-1, int_b=1, wmat=None):
    """ project to a valid moment sequence on interval [a,b]

    Args:
    moments: a sequence of estimated moments, starting from degree 1
    [a,b]: range of distributions, default [-1,1]
    wmat: weighting matrix, default identity matrix

    Returns:
    a sequence of valid moments on [a,b], starting from degree 1 (default [a,b]=[-1,1])
    minimize (moments-x)' * wmat * (moments-x), subject to x is a valid moment sequence
    """
    length = len(moments)
    if length == 0:
        return moments
    if length == 1:
        moments[0] = max(int_a, min(int_b, moments[0]))
        return moments


    variables = cvxpy.Variable(length) # variables [m_1,m_2,...,m_n]
    if wmat is None:
        wmat = np.identity(length)
    obj = cvxpy.Minimize(cvxpy.quad_form(moments-variables, wmat)) # objective function
    # obj = cvxpy.Minimize(cvxpy.sum_squares(x - moments)) 

    # the following gives constraints
    # Ref for PSD condition: [Lasserre 2009, Theorem 3.3 and 3.4]
    if length % 2 == 1:
        # odd case
        k = int((length+1)/2)
        h_mat = cvxpy.Variable(k, k+1)
        constraints = [h_mat[:, 1:]-int_a*h_mat[:, :k]>>0,
                       int_b*h_mat[:, :k]-h_mat[:, 1:]>>0]
    else:
        # even case
        k = int(length/2)+1
        h_mat = cvxpy.Variable(k, k)
        constraints = [h_mat>>0,
                       (int_a+int_b)*h_mat[:k-1, 1:]-int_a*int_b*h_mat[:k-1, :k-1]+h_mat[1:, 1:]>>0]
    num_row, num_col = h_mat.size
    for i in range(num_row):
        for j in range(num_col):
            if i == 0 and j == 0:
                constraints.append(h_mat[0, 0] == 1)
            else:
                constraints.append(h_mat[i, j] == variables[i+j-1])

    prob = cvxpy.Problem(obj, constraints)
    prob.solve(solver=cvxpy.CVXOPT)
    # opt = prob.solve(solver=cvxpy.CVXOPT)

    return np.asarray(variables.value).reshape(moments.shape)




def quadmom(moments, dettol=0, inf=1e10):
    """ compute quadrature from moments
    ref: Gene Golub, John Welsch, Calculation of Gaussian Quadrature Rules

    Args:
    m: moments sequence
    dettol: tolerant of singularity of moments sequence (quantified by determinant of moment matrix)
    INF: infinity

    Returns:
    U: quadrature
    """

    moments = np.asarray(moments)
    # INF = float('inf')
    inf = 1e10

    if len(moments) % 2 == 1:
        moments = np.append(moments, inf)
    num = int(len(moments)/2)
    moments = np.insert(moments, 0, 1)


    h_mat = hankel(moments[:num+1:], moments[num::]) # Hankel matrix
    for i in range(len(h_mat)):
        # check positive definite and decide to use how many moments
        if np.linalg.det(h_mat[0:i+1, 0:i+1]) <= dettol: # alternative: less than some threshold
            h_mat = h_mat[0:i+1, 0:i+1]
            h_mat[i, i] = inf
            num = i
            break
    r_mat = np.transpose(np.linalg.cholesky(h_mat)) # upper triangular Cholesky factor

    # Compute alpha and beta from r, using Golub and Welsch's formula.
    alpha = np.zeros(num)
    alpha[0] = r_mat[0][1] / r_mat[0][0]
    for i in range(1, num):
        alpha[i] = r_mat[i][i+1]/r_mat[i][i] - r_mat[i-1][i]/r_mat[i-1][i-1]

    beta = np.zeros(num-1)
    for i in range(num-1):
        beta[i] = r_mat[i+1][i+1]/r_mat[i][i]

    jacobi = np.diag(alpha, 0) + np.diag(beta, 1) + np.diag(beta, -1)

    eigval, eigvec = np.linalg.eig(jacobi)

    atoms = eigval
    weights = moments[0] * np.power(eigvec[0], 2)

    return DiscreteRV(weights=weights, atoms=atoms)
