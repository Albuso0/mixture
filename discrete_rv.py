"""
module for common operations on discrete random variables (distributions)
"""


import numpy as np

def assert_shape_equal(x_var, y_var):
    """
    Assert shape equal
    """
    if x_var.shape != y_var.shape:
        raise AssertionError('Shape mismatch!')


class DiscreteRV:
    """ class for 1-d finite discrete RV
    """
    def __init__(self, w, x):
        """
        weights: probabilites masses
        atoms: atoms
        """
        self.weights = np.asarray(w)
        self.atoms = np.asarray(x)
        assert_shape_equal(self.weights, self.atoms)

    def __repr__(self):
        return "atom: %s\nwght: %s" % (self.atoms, self.weights)

    def moment(self, degree=1):
        """ Compute the moments of the input RV up to the given degree (start from first degree)
        Args:
        degree: int
        highest degree k

        Returns:
        array of moments from the first degree to degree k
        """
        moments = np.zeros(degree)
        monomial = np.ones(self.atoms.shape)

        for i in range(degree):
            monomial *= self.atoms
            moments[i] = np.dot(self.weights, monomial)

        return moments

    def dist_w1(self, another_rv):
        """
        Compute the W1 distance from another_rv
        """
        return wass(self, another_rv)


    def sample(self, num):
        """ draw n iid sample from U
        library: https://docs.scipy.org/doc/numpy/reference/routines.random.html
        alternative way: scipy.stats.rv_discrete(name='custm', values=(xk, pk))
        """
        ## for 1-d RV
        return np.random.choice(self.atoms, size=num, replace=True, p=self.weights)

        ## for U in higher dimensions
        # return U.x[np.random.choice(U.x.shape[0],size=n, replace=True, p=U.p)]



    def sample_noisy(self, num, sigma=1):
        """ draw n iid samples from model U+sigma Z. Default sigma=1.
        """
        return self.sample(num) + sigma*np.random.randn(num)

def wass(u_rv, v_rv):
    """ compute W1 distance between DiscreteRVs U and V
    """
    if len(u_rv.atoms) == 0 or len(v_rv.atoms) == 0:
        return 0.

    x_u, p_u = zip(*sorted(zip(u_rv.atoms, u_rv.weights)))
    x_v, p_v = zip(*sorted(zip(v_rv.atoms, v_rv.weights)))
    l_u, l_v, diff_cdf, dist, pre = 0, 0, 0., 0., 0.
    while l_u < len(x_u) or l_v < len(x_v):
        if l_v == len(x_v) or (l_u < len(x_u) and x_v[l_v] > x_u[l_u]):
            dist += abs(diff_cdf)*(x_u[l_u]-pre)
            pre = x_u[l_u]
            diff_cdf += p_u[l_u]
            l_u += 1
        else:
            dist += abs(diff_cdf)*(x_v[l_v]-pre)
            pre = x_v[l_v]
            diff_cdf -= p_v[l_v]
            l_v += 1

    return dist
