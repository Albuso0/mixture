import numpy as np

def assert_shape_equal(x,y):
    if x.shape != y.shape:
        raise AssertionError('Shape mismatch!')


class finiteRV:
    """ class for 1-d finite discrete RV
    p: probabilites
    x: atoms
    """
    def __init__(self, prob=[], val=[]):
        self.p = np.asarray(prob)
        self.x = np.asarray(val)
        assert_shape_equal(self.p, self.x)



def moment(RV, degree=1):
    """ Compute the moments of the input RV up to the given degree (start from first degree)
    """
    M = np.zeros(degree)
    monomial = np.ones(RV.x.shape)

    for i in range(degree):
        monomial *= RV.x
        M[i] = np.dot(RV.p, monomial)
        
    return M



def W1(U,V):
    """ compute W1 distance between finiteRVs U and V
    """
    if len(U.x)==0 or len(V.x)==0:
        return 0.
    
    x1, p1 = zip(*sorted(zip(U.x, U.p)))
    x2, p2 = zip(*sorted(zip(V.x, V.p)))
    l1,l2,diffCDF,dist,pre = 0,0,0.,0.,0.
    while l1 < len(x1) or l2 < len(x2):
        if l2==len(x2) or (l1<len(x1) and x2[l2] > x1[l1]):
            dist += abs(diffCDF)*(x1[l1]-pre)
            pre = x1[l1]
            diffCDF += p1[l1]
            l1 += 1
        else:
            dist += abs(diffCDF)*(x2[l2]-pre)
            pre = x2[l2]
            diffCDF -= p2[l2]
            l2 += 1

    return dist





def sample(U,n):
    """ draw n iid sample from U
    library: https://docs.scipy.org/doc/numpy/reference/routines.random.html
    alternative way: scipy.stats.rv_discrete(name='custm', values=(xk, pk)) 
    """
    ## for 1-d RV
    return np.random.choice(U.x, size=n, replace=True, p=U.p)

    ## for U in higher dimensions
    # return U.x[np.random.choice(U.x.shape[0],size=n, replace=True, p=U.p)] 



def sampleNoisy(U,n,sigma=1):
    """ draw n iid samples from model U+sigma Z. Default sigma=1.
    """
    return sample(U,n) + sigma*np.random.randn(n)

    
