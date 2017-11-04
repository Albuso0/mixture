"""
main module
"""
import numpy as np
import moments as mm

class DMM():
    """
    class for denoised method of moments
    """
    def __init__(self, k, a=-10, b=10, sigma=None):
        """
        Args:
        sigma: float, default None
        standard deviation. If sigma == None, will estimate sigma.

        a, b: floats, default -10 and 10
        represents the interval of means [a,b]

        num_components: int, required
        number of postulated components
        """
        self.sigma = sigma
        self.left = a
        self.right = b
        self.k = k

    def estimate(self, samples):
        """
        estimate a model from given samples

        Args:
        samples: array of float, required
        samples collected

        Returns:
        latent distribtuion
        """
        samples = np.asarray(samples)
        m_raw = mm.empirical_moment(samples, 2*self.k)

        if self.sigma != None:
            m_raw = m_raw[:2*self.k-1, :]
            tran_mat, tran_b = mm.hermite_transform_matrix(2*self.k-1, self.sigma)
            m_hermite = np.dot(tran_mat, m_raw)+tran_b
            m_decon = np.mean(m_hermite, axis=1)
            # preliminary estimate
            m_proj = mm.projection(m_decon, self.left, self.right)
            dmom_rv = mm.quadmom(m_proj, dettol=0)
            # second step estimate
            # wmat = self.estimate_weight_matrix(samples, dmom_rv)

        return dmom_rv

    def estimate_weight_matrix(self, samples, model):
        """
        estimate weight matrix
        """
        return 0
        
    def estimate_from_moments(self, moments, wmat=None):
        """
        estimate a model from given moments
        """
        return 0

    # input: samples
    # output: estimated model
    # provide some optional parameters


    # known variance
    ## given weighting matrix
    ## no weighting matrix
    ### 1. use identity weighting matrix
    ### 2. consistent estimation of optimal weighting matrix
    ### 3. re-estimate


    # unknown variance
