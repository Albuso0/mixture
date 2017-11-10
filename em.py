"""
module for EM algorithm in Gaussian mixture model
"""
import numpy as np
from model_gm import ModelGM

class EM():
    """
    class for EM algorithm
    """
    def __init__(self, k, sigma=None, tol=1e-3, max_iter=100, print_iter=False):
        """
        Args:
        tol: float
        stop criterion

        k: int
        number of components

        sigma: common standard deviation
        if sigma is None, it will be estimated

        tol: float
        stop criterion, tolerance of log-likelihood increase

        max_iter: int
        maximum number of interations

        print_iter: bool
        if True, will print more information on each iteration
        """
        self.k = k
        self.sigma = sigma
        self.tol = tol
        self.max_iter = max_iter
        self.print_iter = print_iter

    def estimate(self, samples, num_rd=5, x_range=None, s_range=None):
        """
        estimate a model
        best of num_rd random initial guesses
        initial means are uniform from x_range
        initial sigma are uniform from s_range

        Args:
        num_rd(int): number of random initial guess, default 5
        x_range [a,b]: initial guess range of means, default [-1,1]
        s_range [c,d]: initial guess range of sigmas, default [0.5,1.5]
        """
        if x_range is None:
            x_range = [-1, 1]
        if s_range is None:
            s_range = [0.5, 1.5]

        ll_max = float('-inf')
        for _ in range(num_rd):
            w_init = np.random.dirichlet(np.ones(self.k))
            x_init = np.random.uniform(x_range[0], x_range[1], self.k)
            if self.sigma is None:
                s_init = np.random.uniform(s_range[0], s_range[1])
            else:
                s_init = self.sigma
            start = ModelGM(w=w_init, x=x_init, std=s_init)
            model_cur, _, ll_cur = self.estimate_with_init(samples, start, detail=True)
            if ll_cur > ll_max:
                model = model_cur
                ll_max = ll_cur
        return model

    def estimate_with_init(self, samples, init, detail=False):
        """
        estimate a model from a given initial
        Args:
        init (modelGM): initial guess

        Returns:
        model(modelGM): estimated model
        iterN(int): number of iterations
        ll_cur(float): last log-likelihood
        """
        # assert self.k == len(init.weights)
        k_cur = len(init.weights)
        num = len(samples)
        samples = np.asarray(samples)

        num_iter = 0
        model = init
        l_mat = np.exp(ll_mat(samples, model)) # shape (n,k)
        ll_cur = np.sum(np.log(np.dot(l_mat, model.weights)))

        while True:
            ll_pre = ll_cur

            labels = l_mat * model.weights # shape (n,k)
            labels /= np.sum(labels, axis=1)[:, np.newaxis]

            sum_labels = np.sum(labels, axis=0) # shape (k,)

            num_iter += 1
            model.weights = sum_labels/num
            model.centers = np.dot(samples, labels)/sum_labels

            if self.sigma is None:
                # EM iteration of estimating the common variance
                cross = model.centers**2-2*np.outer(samples, model.centers)\
                        +(samples**2)[:, np.newaxis]
                sigma2 = np.sum(cross*labels)/num
                model.sigma = np.ones(k_cur) * np.sqrt(sigma2)

            l_mat = np.exp(ll_mat(samples, model))
            ll_cur = np.sum(np.log(np.dot(l_mat, model.weights)))

            if self.print_iter:
                print(model.weights, model.centers, model.sigma)
                print(ll_cur)

            if num_iter > self.max_iter or ll_cur-ll_pre < self.tol:
                break

        if detail:
            return model, num_iter, ll_cur
        else:
            return model

def ll_mat(samples, model):
    """
    log-likelihood of samples

    Args:
    samples: ndarray of length n
    model: ModelGM instance of k components with common sigma

    Returns:
    matrix of log-likelihoods of shape (n,k)
    """
    samples = np.asarray(samples)
    precision = 1./(model.sigma**2) # inverse of variance (sigma^2)
    ll0 = model.centers**2*precision - 2*np.outer(samples, model.centers*precision) \
          +np.outer(samples**2, precision)
    return -0.5*(np.log(2*np.pi)+ll0)-np.log(model.sigma)


def ll_sample(samples, model):
    """
    Log-likelihood matrix of samples under the given GM model

    Args:
    samples: ndarray of length n
    model: ModelGM instance of k components with common sigma

    Return:
    log-likelihood of all samples
    """
    return np.sum(np.log(np.dot(np.exp(ll_mat(samples, model)), model.weights)))
