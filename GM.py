from discreteRV import *

class modelGM:
    """ class for parameters in 1-d GM model
    p: weights
    mean: means
    sigma: standard deviations
    """
    def __init__(self, prob=1, mean=0, std=1):
        """ Initialize a GM model
        prob: arrary of weights
        mean: arrary of means
        std: scalar or array of standard deviations. If scalar, components share the same std.
        """

        self.p = np.asarray(prob)
        self.mu = np.asarray(mean)
        assert_shape_equal(self.p, self.mu)
        # np.testing.assert_array_equal(self.p.shape, self.mu.shape)
        
        if np.isscalar(std):
            self.sigma = std*np.ones(self.p.shape)
        else:
            self.sigma = np.asarray(std)
        assert_shape_equal(self.p, self.sigma)

    def meanRV():
        return finiteRV(self.p,self.mu)

    def stdRV():
        return finiteRV(self.p,self.sigma)



def sampleGM(GM,n):
    idx = np.random.choice(GM.mu.shape[0], size=n, replace=True, p=GM.p)
    return GM.mu[idx] + GM.sigma[idx] * np.random.randn(n)
