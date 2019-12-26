Learning a Gaussian location mixture
========
Denoised Method of Moments (DMM) implementation in [Python3](https://docs.python.org/3/using/index.html).


Denoised Method of Moments (DMM)
======

External dependencies
----
* [Numpy and SciPy](https://www.scipy.org)
* [CVXPY](http://www.cvxpy.org) (version >= 1.0)
* [CVXOPT](http://cvxopt.org)


Basic script
----
```python
>>> from dmm import *
>>> x = np.random.choice([0,1],10000)+np.random.randn(10000) # 10,000 samples from uniform mixture of N(0,1) and N(1,1)
>>> dmm = DMM(k=2) # DMM estimator for two components with unknown variance
>>> dmm.estimate(x)
atom: [ 1.0196572  -0.06853119]
wght: [ 0.51403795  0.48596205]
sigm: 0.97182312069
>>> dmm = DMM(k=2, sigma=1) # DMM estimator for two components with given standard deviation
>>> dmm.estimate(x)
atom: [ 0.96256605 -0.01854031]
wght: [ 0.51917043  0.48082957]
sigm: 1.0
```

More on DMM
----
* `dmm.estimate(samples)` uses two-step method: 
  1. prelimnary estimation with identity weight matrix, and (consistent) estimation of optimal weight matrix (the same procedure as GMM, [ref: [Bruce E. Hansen] Econometrics. Chapter 11](https://www.ssc.wisc.edu/~bhansen/econometrics/Econometrics.pdf)); 
  2. reestimate parameters using esimated weight matrix.
* `dmm.estimate_online(new_samples)` instead of accessing all n samples, this method only stores moments estimate of size O(k), and sample correlation matrix of size O(k*k). 
  - The weight matrix is the inverse of sample covariance matrix of moments estimate (another consistent estimation of optimal weight matrix). 
* `dmm.estimate_with_wmat(samples, wmat)` estimate with user-specified weight matrix. 
  - Only usable with given standard deviation.
  - Default: identity matrix.
* `dmm.estimate_select(samples, threhold)` estimate with possibly fewer number of components. This methods only uses moments estimate whose sample variance is less than n*threhold.
  - Default threhold=1. 


Other methods
=========

Expectation–Maximization (EM)
-----
EM algorithm with either given sigma or unknown sigma is implemented.

```python
>>> from em import *
>>> em = EM(k=2) # EM estimator for two components with unknown variance
>>> em.estimate(x)
atom: [-0.09813523  0.98464457]
wght: [ 0.45605361  0.54394639]
sigm: 0.974374957959
>>> em = EM(k=2,sigma=1) # EM estimator for two components with given standard deviation
>>> em.estimate(x)
atom: [  9.85003578e-01   3.80442325e-04]
wght: [ 0.49811799  0.50188201]
sigm: 1.0
```


The usual Method of Moments (MM)
-----
Method of Moments with given sigma is implemented. 
Extra dependency: [Sympy](http://www.sympy.org). 

```python
>>> from mm import *
>>> mm = MM(k=2,sigma=1)
>>> mm.estimate(x)
atom: [-0.0184787563625268 0.962539250975350]
wght: [0.480827376053319 0.519172623946681]
sigm: 1.0
```

Generalized Method of Moments (GMM)
-----
This is implemented in [R](https://www.r-project.org).
Extra dependencies:
* [gmm](https://cran.r-project.org/web/packages/gmm/index.html) 
* the pseudo-random number generators in [MCMCpack](https://cran.r-project.org/web/packages/MCMCpack/index.html) 

```R
> source('gmmGM.R')
> x <- sample(c(0,1), 10000, replace=T, prob=c(0.5,0.5))+rnorm(10000) ## 10,000 samples from uniform mixture of N(0,1) and N(1,1)
> gmmGM(k=2,x) # GMM estimate for two components with unknown variance
$Weights
[1] 0.507491 0.492509

$Centers
[1] 0.02134997 1.00139903

$Sigma
[1] 0.9839962

> gmmGM(k=2,x,sigma=1) # GMM estimate for two components with given standard deviation
$Weights
[1] 0.5092649 0.4907351

$Centers
[1] 0.05599395 0.96899030

$Sigma
[1] 1

```


Useful tools for synthetic evaluation
======
Random samples generation
-----
* In Python, this is implemented by `sample_gm` in *model_gm.py*;
* In R, this is implemented by `sampleGM` in *gmmGM.R*. 

### Script in Python
```python
>>> from model_gm import *
>>> model = ModelGM(w=[0.5, 0.5], x=[0, 1], std=1) # mixing weights w; centers x; sigma=std
>>> sample = sample_gm(model, 10000)
```

### Script in R
```R
> sample = sampleGM(10000, p=c(0.5, 0.5), x=c(0, 1), sigma=1)
```

W<sub>1</sub> distance evaluation
----
* In Python, this is implemented by `wass` in *discrete_rv.py*;
* In R, this is implemented by `w1` in *gmmGM.R*. 


### Script in Python
```python
>>> from discrete_rv import *
>>> esti = dmm.estimate(x)
>>> wass(esti.mean_rv(),model.mean_rv())
0.057446052574283327
```

### Script in R
```R
> esti <- gmmGM(k=2, x, sigma=1)
> w1(c(0.5,0.5), c(0,1), esti[[1]], esti[[2]])
[1] 0.05196058
```
