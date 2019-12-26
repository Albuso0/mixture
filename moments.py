"""
module for operations on moments
"""

import numpy as np
import cvxpy
from scipy.linalg import hankel
from discrete_rv import DiscreteRV
import warnings

def empirical_moment(samples, degree):
    """ Compute empirical moments of samples
    Args:
    samples: x=(x1...xn)
    degree: L

    Returns
    matrix M of size L*n
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


def transform(mat, x_var):
    """
    Compute a linear tranformation Ax+b

    mat: Tuple (A,b)
    A: matrix of shape (k,k)
    b: matrix of shape (k,1)

    x_var: variable x
    array of length k

    Returns:
    array of length k, y=Ax+b
    """
    k = len(mat)
    x_var = x_var.reshape((k, 1))
    y_var = np.dot(mat[0], x_var)+mat[1]
    return y_var.reshape(k)



def projection(moments, interval=None, wmat=None):
    """ project to a valid moment sequence on interval [a,b]

    Args:
    moments: a sequence of estimated moments, starting from degree 1
    interval [a,b]: range of distributions, default [-1,1]
    wmat: weighting matrix, default identity matrix

    Returns:
    a sequence of valid moments on [a,b], starting from degree 1 (default [a,b]=[-1,1])
    minimize (moments-x)' * wmat * (moments-x), subject to x is a valid moment sequence
    """
    if interval is None:
        interval = [-1, 1]

    length = len(moments)
    if length == 0:
        return moments
    if length == 1:
        moments[0] = max(interval[0], min(interval[1], moments[0]))
        return moments


    # preliminary filtering of moments based on range
    r_max = max(abs(interval[0]),abs(interval[1]))
    m_max = 1
    for i in range(len(moments)):
        m_max *= r_max
        if moments[i] > m_max:
            moments[i] = m_max
        elif moments[i] < -m_max:
            moments[i] = -m_max    

    # SDP for further projection
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
        h_mat = cvxpy.Variable((k, k+1))
        constraints = [h_mat[:, 1:]-interval[0]*h_mat[:, :k]>>0,
                       interval[1]*h_mat[:, :k]-h_mat[:, 1:]>>0]
    else:
        # even case
        k = int(length/2)+1
        h_mat = cvxpy.Variable((k, k))
        constraints = [h_mat>>0,
                       (interval[0]+interval[1])*h_mat[:k-1, 1:]-interval[0]*interval[1]*h_mat[:k-1, :k-1]-h_mat[1:, 1:]>>0]
    num_row, num_col = h_mat.shape
    for i in range(num_row):
        for j in range(num_col):
            if i == 0 and j == 0:
                constraints.append(h_mat[0, 0] == 1)
            else:
                constraints.append(h_mat[i, j] == variables[i+j-1])

    prob = cvxpy.Problem(obj, constraints)
    try:
        prob.solve(solver=cvxpy.CVXOPT)
    except Exception as e:
        warnings.warn("CVXOPT failed. Using SCS solver..."+str(e))
        prob.solve(solver=cvxpy.SCS)
        # prob.solve()
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

    return DiscreteRV(w=weights, x=atoms)


def deconvolve_unknown_variance(moments):
    """ Deconvolution with unknown sigma, using Lindsay's estimator.
    Fit moments with U+sigma Z. Estimate common sigma, and moments of U.

    Args:
    moments: array of float, length 2k
    moments estimate of degree 1 to 2k

    Returns:
    Tuple of deconvolved moments and estimated variance (sigma^2)
    """

    moments = np.insert(moments, 0, 1)

    length = len(moments)
    m_hermite = [0]*length
    x_var = np.poly1d([1, 0]) # x = sigma^2

    if length > 0:
        prepre = np.zeros(moments.shape)
        prepre[0]=1
        m_hermite[0] = moments[0]
    if length > 1:
        pre = np.zeros(moments.shape)
        pre[1]=1
        m_hermite[1] = moments[1]
    for k in range(2, length):
        # recursion: H_{n+1}(x) = x * H_n(x) - n * H_{n-1}(x)
        coeffs = np.roll(pre, 1) - prepre*(k-1)
        for i in range(k+1):
            m_hermite[k] += float(coeffs[i]*moments[i])*(x_var**(int((k-i)/2)))
        prepre = pre
        pre = coeffs

    # Solve the first non-negative root
    equation = det_mom(m_hermite)

    root = equation.r
    root = root[np.isreal(root)].real
    root = root[root >= 0]
    root.sort()
    # print(root)
    root0 = root[0]

    for k in range(2, length):
        m_hermite[k] = m_hermite[k](root0)

    return (np.asarray(m_hermite[1:]), float(root0))



def det_mom(moments):
    """
    Compute determinant of moment matrix

    Args:
    moments: array of moments of

    Return:
    determinant of moment matrix
    """
    return determinant(get_hankel(moments))


def get_hankel(moments):
    """
    Construct Hankel matrix from (m_0,...,m_{2k})

    Args:
    moments: array of floats of length 2k+1
    moments of degrees from 0 to 2k

    Return:
    Hankel matrix from those moments of size (k+1,k+1)
    """
    # length of m = 2k+1 = 2k_inc-1
    k_inc = int((len(moments)+1)/2)
    matrix = [[0]*k_inc for i in range(k_inc)]
    for i in range(k_inc):
        for j in range(k_inc):
            matrix[i][j] = moments[i+j]

    return matrix


def determinant(mat, rows=None, cols=None):
    """
    Compute the determinant of a submatrix

    Args:
    mat: matrix, list of lists
    input matrix

    rows: list of int
    selection of rows

    cols: list of int, same size as rows
    selection of columns

    Returns:
    determinant of submatrix mat[rows, cols]
    """
    if rows is None:
        num_rows, num_cols = len(mat), len(mat[0])
        rows = list(range(num_rows))
        cols = list(range(num_cols))

    num_rows = len(rows)
    num_cols = len(cols)
    assert num_rows == num_cols

    if num_rows == 1:
        return mat[rows[0]][cols[0]]
    # elif rows == 2:
    #     return M[r[0]][c[0]]*M[r[1]][c[1]] - M[r[0]][c[1]]*M[r[1]][c[0]]
    # elif rows == 3:
    #     return (M[r[0]][c[0]]*M[r[1]][c[1]]*M[r[2]][c[2]]
    #             + M[r[0]][c[1]]*M[r[1]][c[2]]*M[r[2]][c[0]]
    #             + M[r[0]][c[2]]*M[r[1]][c[0]]*M[r[2]][c[1]])
    #             - (M[r[0]][c[2]]*M[r[1]][c[1]]*M[r[2]][c[0]]
    #             + M[r[0]][c[0]]*M[r[1]][c[2]]*M[r[2]][c[1]]
    #             + M[r[0]][c[1]]*M[r[1]][c[0]]*M[r[2]][c[2]])
    else:
        det = 0
        newr = rows[1:]
        sign = 1
        for k in range(num_cols):
            newc = cols[:k] + cols[(k + 1):]
            det += determinant(mat, newr, newc)*mat[rows[0]][cols[k]]*sign
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
