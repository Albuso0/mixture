"""
main module
"""
import numpy as np
import moments as mm
from model_gm import ModelGM

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
    def __init__(self, k, interval=None, sigma=None):
        """
        Args:
        sigma: float, default None
        standard deviation. If sigma == None, will estimate sigma.

        interval =[a,b]: floats, default [-10, 10]
        represents the interval of means [a,b]

        num_components: int, required
        number of postulated components
        """
        if interval is None:
            interval = [-10, 10]

        self.sigma = sigma
        self.k = k
        self.interval = interval

        if self.sigma is None:
            self.num_mom = 2*self.k
        else:
            self.num_mom = 2*self.k-1
            self.transform = mm.hermite_transform_matrix(2*self.k-1, self.sigma)

        # for online estimation
        # if sigma is None, this stores list of (sample size, raw moments)
        # if sigma is known, this store list of (sample size, Hermite moments, correlation)
        if self.sigma is None:
            self.online = [0, np.zeros((self.num_mom, 1))]
        else:
            self.online = [0, np.zeros((self.num_mom, 1)), np.zeros((self.num_mom, self.num_mom))]


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
        an estimated ModelGM
        """
        samples = np.asarray(samples)
        m_raw = mm.empirical_moment(samples, self.num_mom)

        if self.sigma is None:
            m_raw = np.mean(m_raw, axis=1)
            m_esti, var_esti = mm.deconvolve_unknown_variance(m_raw)
            dmom_rv = mm.quadmom(m_esti[:2*self.k-1])
            return ModelGM(w=dmom_rv.weights, x=dmom_rv.atoms, std=np.sqrt(var_esti))
        else:
            m_hermite = np.dot(self.transform[0], m_raw)+self.transform[1]
            m_decon = np.mean(m_hermite, axis=1)
            dmom_rv = self.estimate_from_moments(m_decon) # preliminary estimate
            wmat = estimate_weight_matrix(m_hermite, dmom_rv)
            dmom_rv = self.estimate_from_moments(m_decon, wmat) # second step estimate
            # print(np.linalg.inv(wmat))
            return ModelGM(w=dmom_rv.weights, x=dmom_rv.atoms, std=self.sigma)

    def estimate_online(self, samples_new):
        """
        update the estimate a model from more samples
        only store a few moments and correlations

        Args:
        samples_new: array of floats
        new samples

        Returns:
        an estimated ModelGM
        """
        samples_new = np.asarray(samples_new)
        m_new = mm.empirical_moment(samples_new, self.num_mom) # moments, shape (L,n)
        n_new = len(samples_new)
        n_total = self.online[0]+n_new

        if self.sigma:
            m_new = np.dot(self.transform[0], m_new)+self.transform[1]
            cor_new = np.dot(m_new, m_new.T)/n_new
            self.online[2] = self.online[2]*(self.online[0]/n_total)+cor_new*(n_new/n_total)

        mom_new = np.mean(m_new, axis=1)[:, np.newaxis] # empirical moments, shape (L,1)
        self.online[1] = self.online[1]*(self.online[0]/n_total)+mom_new*(n_new/n_total)
        self.online[0] = n_total

        if self.sigma is None:
            m_esti, var_esti = mm.deconvolve_unknown_variance(self.online[1])
            dmom_rv = mm.quadmom(m_esti[:2*self.k-1])
            return ModelGM(w=dmom_rv.weights, x=dmom_rv.atoms, std=np.sqrt(var_esti))
        else:
            wmat = np.linalg.inv(self.online[2]-np.dot(self.online[1], self.online[1].T))
            dmom_rv = self.estimate_from_moments(self.online[1].reshape(self.num_mom), wmat)
            # print(np.linalg.inv(wmat))
            return ModelGM(w=dmom_rv.weights, x=dmom_rv.atoms, std=self.sigma)

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
        assert self.sigma

        m_latent = self.estimate_latent_moments(samples)
        dmom_rv = self.estimate_from_moments(m_latent, wmat)
        return ModelGM(w=dmom_rv.weights, x=dmom_rv.atoms, std=self.sigma)

    def estimate_select(self, samples, threhold=1):
        """
        estimate with selected number of components
        """
        k_cur = min(self.select_num_comp(samples, threhold), self.k)
        dmm_cur = DMM(k=k_cur, interval=self.interval, sigma=self.sigma)
        return dmm_cur.estimate(samples)

    def estimate_latent_moments(self, samples):
        """
        estimate moments of latent distribution (deconvolution)
        model: X=U+sigma*Z
        sigma must be given

        Args:
        samples: array of length n

        Return:
        array of length 2k-1
        estimated moments of U from degree 1 to 2k-1
        """
        assert self.sigma

        samples = np.asarray(samples)
        m_raw = mm.empirical_moment(samples, self.num_mom)
        m_raw = np.mean(m_raw, axis=1).reshape((self.num_mom, 1))
        return ((np.dot(self.transform[0], m_raw)+self.transform[1])).reshape(self.num_mom)

    def estimate_from_moments(self, moments, wmat=None):
        """
        estimate a discrete random variable from moments estimate

        Args:
        moments: array of length 2k-1
        estimated moments of U of degree 1 to 2k-1

        wmat: matrix of shape (k, k)
        weight matrix for moment projection, default identity matrix

        Returns:
        an estimated latent distribtuion on at most k points
        """
        m_proj = mm.projection(moments, self.interval, wmat)
        dmom_rv = mm.quadmom(m_proj, dettol=0)
        return dmom_rv

    def sample_moment_cov(self, samples):
        """
        return the sample covariance matrix of moments estimates
        """
        samples = np.asarray(samples)
        mom = mm.empirical_moment(samples, self.num_mom) # moments, shape (L,n)
        mean = np.mean(mom, axis=1)
        num = len(samples)
        cor = np.dot(mom, mom.T)/num - np.outer(mean, mean)
        return cor

    def select_num_comp(self, samples, threhold=1):
        """
        select the number of components
        according to sample variance of moments estimate
        """
        samples = np.asarray(samples)
        num = len(samples)

        m_raw = np.ones(len(samples))
        deg_cur = 0
        while True:
            deg_cur += 1
            m_raw *= samples
            var = np.mean(m_raw**2)-np.mean(m_raw)**2
            if var > threhold*num:
                break

        # moments of degree 1 to (deg_cur-1) is accurate
        if self.sigma:
            return int(np.floor(deg_cur/2))
        else:
            return int(np.floor((deg_cur-1)/2))

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
