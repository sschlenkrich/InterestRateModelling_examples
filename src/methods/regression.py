
import numpy as np
from scipy.linalg import lstsq

def multi_index_set(n, k):
    """Polynomial degrees for n variables up to order k-1."""
    if n==1: return [ [i] for i in range(k) ]
    return [ [i]+s for i in range(k) for s in multi_index_set(n-1,k-i)]

class Regression:

    # Python constructor
    def __init__(self, controls, observations, max_polynomial_degree=2):
        self.max_polynomial_degree = max_polynomial_degree
        self.multi_idx_set = np.array(multi_index_set(controls.shape[0],max_polynomial_degree+1))
        A = self.monomials(controls).T
        p, res, rnk, s = lstsq(A, observations)   # res, rnk, s for debug purposes
        self.beta = p

    def monomials(self, control):
        if len(control.shape)==1:  # control is a vector
            x = np.ones(self.multi_idx_set.shape[0])
        if len(control.shape)==2:  # control is a matrix; multi-path
            x = np.ones([self.multi_idx_set.shape[0], control.shape[1]])
        for i in range(self.multi_idx_set.shape[0]):
            for j in range(self.multi_idx_set.shape[1]):
                x[i] *= control[j]**self.multi_idx_set[i,j]
        return x

    def value(self, control):
        return self.beta.dot(self.monomials(control))
