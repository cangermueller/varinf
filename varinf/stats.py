import numpy as np
import scipy.stats
from scipy.special import multigammaln


class Wishart(object):
    def __init__(self, W, v):
        self.d = W.shape[0]
        if W.shape != (self.d, self.d):
            raise TypeError('W must be a square matrix!')
        self.W = W
        self.v = v
        self.logC = -(0.5*self.v*self.d*np.log(2)+\
                      0.5*self.v*np.log(np.linalg.det(self.W))+\
                      multigammaln(0.5*v, self.d))
        self.Winv = np.linalg.inv(W)

    def pdf(self, X):
        return np.exp(self.logC+\
                      0.5*(self.v-self.d-1)*np.log(np.linalg.det(X))-\
                      0.5*np.trace(self.Winv.dot(X)))


