
import numpy as np

from src.methods.regression import Regression

class StateVariableControls:
    """
    Return model state variables as controls for regression.
    """
    def __call__(self, X, t):
        """
        Calculate regression variables.

        X  ... 2d array of shape (n_states, n_paths)
        t  ... observation time of X

        Returns:
           2d array of shape (n_controls, n_paths)
        """
        return X[0].reshape(1,-1)

class CoterminalRateControls:
    """
    Return co-terminal rate as controls for regression.
    """
    def __init__(self, model, maturity, strike_rate=None):
        self.model = model
        self.maturity = maturity
        self.strike_rate = strike_rate

    def __call__(self, X, t):
        """
        Calculate regression variables.

        X  ... 2d array of shape (n_states, n_paths)
        t  ... observation time of X

        Returns:
           2d array of shape (n_controls, n_paths)
        """
        zb = self.model.zero_bond(t, X[0], self.maturity)
        rate = -np.log(zb) / (self.maturity - t)
        if self.strike_rate is not None:
            return np.array([ rate, np.maximum(rate - self.strike_rate, 0.0) ])
        else:
            return rate.reshape(1,-1)


class AmcSolver:

    # Python constructor
    def __init__(self, simulation, max_polynomial_degree=2, split_ratio=0.25, controls=StateVariableControls()):
        self.simulation = simulation
        self.max_polynomial_degree = max_polynomial_degree
        self.minSampleIdx = int(split_ratio*self.simulation.n_paths)  # we split training data and simulation data
        self.controls = controls

    def nearest_index(self, t):
        idx = np.searchsorted(self.simulation.times, t)
        if idx > 0 and \
           (idx == len(self.simulation.times) or \
            np.abs(t - self.simulation.times[idx-1]) < np.abs(t - self.simulation.times[idx])
           ):
            return idx-1
        else:
            return idx

    def states(self,expiryTime):
        t_idx = self.nearest_index(expiryTime) # assume simulation fits observation times
        return self.simulation.X[t_idx,:,:]

    def roll_back(self, T0, T1, x1, U1, H1):        
        x0 = self.states(T0)
        N0 = self.simulation.model.numeraire(x0, T0)
        N1 = self.simulation.model.numeraire(x1, T1)
        if self.minSampleIdx>0 and T0>0:   # do not use regression for the last roll-back
            C = self.controls(x0[:,:self.minSampleIdx], T0)
            O = N0[:self.minSampleIdx] / N1[:self.minSampleIdx] * \
                np.maximum(U1[:self.minSampleIdx],H1[:self.minSampleIdx])
            R = Regression(C,O,self.max_polynomial_degree)
            V0 = R.value(self.controls(x0, T0))
        else:
            V0 = N0 / N1 * np.maximum(U1, H1)
        if T0==0: 
            sampleIdx = self.minSampleIdx if self.minSampleIdx<self.simulation.n_paths else 0
            return ( np.zeros((1,)), np.mean(V0[sampleIdx:], keepdims=True) )
        return (x0, V0)


class AmcSolverOnlyExerciseRegression(AmcSolver):

    # Python constructor
    def __init__(self, simulation, max_polynomial_degree=2, split_ratio=0.25, controls=StateVariableControls()):
        AmcSolver.__init__(self,simulation,max_polynomial_degree,split_ratio)

    def roll_back(self, T0, T1, x1, U1, H1):        
        x0 = self.states(T0)
        N0 = self.simulation.model.numeraire(x0, T0)
        N1 = self.simulation.model.numeraire(x1, T1)
        if self.minSampleIdx>0 and T0>0:   # do not use regression for the last roll-back
            C = self.controls(x1[:,:self.minSampleIdx], T1)
            O = U1[:self.minSampleIdx] - H1[:self.minSampleIdx]
            R = Regression(C,O,self.max_polynomial_degree)
            I = R.value(self.controls(x1, T1))
        else:
            I = U1 - H1
        V0 = N0 / N1 * ((I>0.0) * U1 + (I<=0.0) * H1)
        if T0==0: 
            sampleIdx = self.minSampleIdx if self.minSampleIdx<self.simulation.n_paths else 0
            return ( np.zeros((1,)), np.mean(V0[sampleIdx:], keepdims=True) )
        return (x0, V0)
