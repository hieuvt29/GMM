from __future__ import division, print_function, unicode_literals
import numpy as np 
from scipy.stats import multivariate_normal as Normal
import sys
from memory_profiler import profile
import scipy 
from scipy.linalg import qr

class NormalDistribution(object):

    def __init__(self, dim , mu = None, sig = None, data = None):
        """
        dim: int
            Number of dimensions.
        mu: array, float
            The mean of the normal distribution.
        sig: array, float
            The covariance matrix or variance of the normal distribution.
        data: array
            Samples or observations used to estimate mu and sigma.
        """
        self.dim = dim
        self.sig = None
        self.mu = None
        self.A = None # precision matrix (inverse matrix of covariance matrix)
        
        if not mu is None:
            assert (type(mu) == np.ndarray or type(mu) == float), "mu must be an ndarray or float"
        if not sig is None:
            assert (type(sig) == np.ndarray or type(sig) == float), "sig must be an ndarray or float"

        # If we have data, create sigma and muy base on data
        if not data is None:
            assert (type(data) == np.ndarray), "data must be an ndarray"
            mu, sig = self.estimate(data, estimate_sig=True)

        # if sigma and muy did not created, we'll create it randomly
        if not sig is not None:
            sig = np.eye(dim)
        if not mu is not None:
            mu = np.random.randn(dim)
        
        self.update(mu = mu, sig = sig)
        
    def update(self, mu = None, sig = None):
        if not mu is None:
            self.mu = mu
        if not sig is None:
            self.sig = sig
            det = 0
            if (self.dim > 1):
                self.A = np.linalg.inv(self.sig)
                det = np.linalg.det(self.sig)
            else:
                self.A = 1/self.sig
                det = np.fabs(self.sig)

            self.factor = ((2 * np.pi) ** (self.dim/2)) * (det ** 0.5)

    def pdf(self, x, method="formular", allow_singular = False):
        """ Normal Distribution Density Function"""
        if (type(x) != np.ndarray and type(self.mu) == np.ndarray ) or (type(x) == np.ndarray and type(self.mu) != np.ndarray ):
            print("Dimension incompatible")
            return None

        if (x.shape != self.mu.shape):
            print("Dimension incompatible: x - {} and mu - {}".format(x.shape, self.mu.shape))
            return None
            
        if method=="formular": 
            A = self.A
            mu = self.mu
            dx = x - mu
            res = np.exp(-0.5 * dx.T.dot(A).dot(dx)) / self.factor

        if method=="lib": res = Normal.pdf(x, mean=self.mu, cov=self.sig, allow_singular=allow_singular)
        
        return res

    def estimate(self, data, estimate_sig = False):
        mu = np.mean(data, axis=0)
        if not estimate_sig:
            sigma = self.sig
        else:
            sigma = np.cov(data, rowvar=0)
        return mu, sigma
    
    def __str__(self):
        return "mu:\n {} \nvariance:\n {}".format(self.mu, self.sig)

# nd = NormalDistribution(2, mu = np.array([1, 1]), sig = np.array([[1, 0], [0, 1]]))

# print(nd.pdf(np.array([1])))

