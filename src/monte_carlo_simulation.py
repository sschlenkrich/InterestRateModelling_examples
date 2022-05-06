
import numpy as np
from tqdm import tqdm

class MonteCarloSimulation:
    """
    Simulate paths for a given diffusion model.
    """

    # Python constructor
    def __init__(self, model, times, n_paths, seed=123, showProgress=False):
        self.model  = model   # an object implementing stochastic process interface
        self.times  = times   # simulation times [0, ..., T], np.array
        self.n_paths = n_paths  # number of paths, long
        # random number generator
        self.dW = np.random.RandomState(seed).standard_normal((len(self.times)-1,model.factors(),self.n_paths))
        # simulate states
        self.X = np.zeros((len(self.times),model.size(),self.n_paths))
        self.X[0] = self.model.initial_values().reshape((-1,1)) * np.ones((1,self.n_paths))
        for i in tqdm(range(len(times)-1), 'Time steps', disable=(not showProgress)):
            self.X[i+1] = model.evolve(self.times[i],self.X[i],times[i+1]-times[i],self.dW[i])
