"""
main module
"""
import numpy as np
import moments as mm


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

        if self.sigma is None:
            self.num_moments = 2*self.k
        else:
            self.num_moments = 2*self.k-1
            self.tran_mat, self.tran_b = mm.hermite_transform_matrix(2*self.k-1, self.sigma)

        # for online estimation
        self.num_samples = 0
        self.moments = np.zeros(self.num_moments)


    def estimate(self, samples):
        """
        estimate a model from given samples
        use two-step estimate:
        1.(a) prelimnary estimation with identity weight matrix
          (b) estimation of optimal weight matrix (require all samples)
        2.    reestimate parameters using esimated weight matrix

        Args:
        samples: array of float, required
        samples collected

        Returns:
        if sigma is given, return estimated latent distribtuion
        if sigma is None, return a tuple of estimated latent distribtuion and estimated sigma
        """
        samples = np.asarray(samples)
        m_raw = mm.empirical_moment(samples, self.num_moments)

        if self.sigma is None:
            m_raw = np.mean(m_raw, axis=1)
            m_esti, var_esti = mm.deconvolve_unknown_variance(m_raw)
            dmom_rv = self.estimate_from_moments(m_esti[:2*self.k-1])
            return dmom_rv, np.sqrt(var_esti)
        else:
            m_hermite = np.dot(self.tran_mat, m_raw)+self.tran_b
            m_decon = np.mean(m_hermite, axis=1)
            # preliminary estimate
            dmom_rv = self.estimate_from_moments(m_decon)
            # second step estimate
            wmat = estimate_weight_matrix(m_hermite, dmom_rv)
            # print(np.linalg.inv(wmat))
            dmom_rv = self.estimate_from_moments(m_decon, wmat)
            return dmom_rv


    def estimate_online(self, samples_new):
        """
        update the estimate a model from more samples
        only store a few moments
        [TODO] online estimation of covariance matrix of moments condition (Kalman filter does?)

        Args:
        samples_new: array of floats
        new samples

        Returns:
        if sigma is given, return estimated latent distribtuion
        if sigma is None, return a tuple of estimated latent distribtuion and estimated sigma
        """
        samples_new = np.asarray(samples_new)
        moments_new = np.mean(mm.empirical_moment(samples_new, self.num_moments), axis=1)
        num_new = len(samples_new)
        num_total = self.num_samples+num_new
        self.moments = self.moments * self.num_samples/num_total + moments_new * num_new/num_total
        self.num_samples = num_total

        if self.sigma is None:
            m_esti, var_esti = mm.deconvolve_unknown_variance(self.moments)
            dmom_rv = self.estimate_from_moments(m_esti[:2*self.k-1])
            return dmom_rv, np.sqrt(var_esti)
        else:
            m_decon = np.dot(self.tran_mat, self.moments)+self.tran_b
            dmom_rv = self.estimate_from_moments(m_decon)
            return dmom_rv


    def estimate_with_wmat(self, samples, wmat=None):
        """
        estimate a model from given samples using given weight matrix
        model: X=U+sigma*Z
        sigma must be given

        Args:
        samples: array of float, required
        samples collected

        wmat: array of shape (k,k)
        weight matrix, default identity

        Returns:
        latent distribtuion
        """
        m_latent = self.estimate_latent_moments(samples)
        return self.estimate_from_moments(m_latent, wmat)


    def estimate_latent_moments(self, samples):
        """
        estimate moments of latent distribution (deconvolution)
        model: X=U+sigma*Z
        sigma must be given

        Args:
        samples: array of length n

        Return:
        estimated moments of U from degree 1 to k
        """
        samples = np.asarray(samples)
        m_raw = mm.empirical_moment(samples, 2*self.k-1)
        m_raw = np.mean(m_raw, axis=1)
        return np.dot(self.tran_mat, m_raw)+self.tran_b

    def estimate_from_moments(self, moments, wmat=None):
        """
        estimate a discrete random variable from moments estimate

        Args:
        moments: array of length 2k-1
        estimated moments of U of degree 1 to 2k-1

        wmat: matrix of shape (k, k)
        weight matrix for moment projection, default identity matrix

        Returns:
        an estimated model on at most k points
        """
        m_proj = mm.projection(moments, self.left, self.right, wmat)
        dmom_rv = mm.quadmom(m_proj, dettol=0)
        return dmom_rv


def estimate_weight_matrix(m_estimate, model):
    """
    estimate weight matrix: inverse of the estimated covariance matrix
    ref: [Bruce E. Hansen] Econometrics. Chapter 11.

    Args:
    m_estimate: matrix of size (k,n)
    power of n samples from degree of 1 to k

    model: discrete_rv

    Return:
    consistent estimation for the optimal weight matrix
    """
    num_moments, num_samples = m_estimate.shape
    mom_model = model.moment(num_moments).reshape((num_moments, 1))
    m_cond = m_estimate - mom_model
    m_cond_avg = np.mean(m_cond, axis=1).reshape((num_moments, 1))
    m_cond_centd = m_cond - m_cond_avg
    return np.linalg.inv(np.dot(m_cond_centd, m_cond_centd.T)/num_samples)
