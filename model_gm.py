"""
module for Gaussian mixture model
"""
from discrete_rv import DiscreteRV, assert_shape_equal
from moments import hermite_transform_matrix, empirical_moment
import numpy as np

class ModelGM:
    """ class for 1-d Gaussian mixture model
    weights: ndarray of weights
    centers: ndarray of means
    sigma: standard deviations
    """
    def __init__(self, w=1, x=0, std=1):
        """ Initialize a GM model
        w: arrary of weights
        x: arrary of means
        std: scalar or array of standard deviations. If scalar, components share the same std.
        """

        self.weights = np.asarray(w)
        self.centers = np.asarray(x)
        assert_shape_equal(self.weights, self.centers)
        # np.testing.assert_array_equal(self.p.shape, self.mu.shape)
        
        if np.isscalar(std):
            self.sigma = std*np.ones(self.weights.shape)
        else:
            self.sigma = np.asarray(std)
        assert_shape_equal(self.weights, self.sigma)

    def __repr__(self):
        return "atom: %s\nwght: %s\nsigm: %s" % (self.centers, self.weights, self.sigma[0])

    def moments_gm(self, degree):
        """
        moments of GM model

        Args:
        degree: int
        highest degree k

        Returns:
        moments of Gaussian mixture model from degree 1 to k
        """
        mom = empirical_moment(self.centers/self.sigma, degree)
        transform = hermite_transform_matrix(degree)
        transform = (np.abs(transform[0]), np.abs(transform[1]))
        mom = np.dot(transform[0], mom) + transform[1]
        s_pow = empirical_moment(self.sigma, degree)
        mom = s_pow*mom
        return np.dot(mom, self.weights)

    def mean_rv(self):
        """
        discrete rv for the means
        """
        return DiscreteRV(self.weights, self.centers)

    def std_rv(self):
        """
        discrete rv for the sigmas
        """
        return DiscreteRV(self.weights, self.sigma)



def sample_gm(model, num):
    """
    n random samples from Gaussian mixture model
    """
    idx = np.random.choice(model.centers.shape[0], size=num, replace=True, p=model.weights)
    return model.centers[idx] + model.sigma[idx] * np.random.randn(num)
