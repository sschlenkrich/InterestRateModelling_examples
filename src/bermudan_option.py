
import numpy as np
from tqdm import tqdm

def bermudan_option_npv(expiry_times, underlying_payoffs, method, showProgress=False):
    """
    Calculate npv of Bermudan option via backward induction algorithm.

    Parameters
    ----------

    expiry_times :
        array/list of option expiry times T_E^1, ..., T_E^k

    underlying_payoffs :
        list of payoff objects, used to calculate U_k(x),
        same length as expiry_times

    method :
        a numerical method object that implements rolling back
        from one expiry time to the next earlier expiry time

    Note:
      - A payoff object needs to implement an 'at(x)' method that
        takes a (all) simulated state variables for a given
        simulation time and returns the corresponding payoff
        values.
      - A state x is assumed a 2d array of shape
        (n_states, n_dimensions). Here, x[0,:] represents
        the Hull White state variable. With MC simulation
        method x[1,:] contains \int_0^t x_s ds.
      - A numerical method object serves two purposes:
        1) Provide a method 'states(T)' that yields all
           modelled/simulated states for a given time T.
        2) Provide a method 'roll_back(...)' that implements
           a single step in backward induction algorithm.
    """
    for k in tqdm(range(len(expiry_times),0,-1), 'Backward induction', disable=(not showProgress)):
        if k==len(expiry_times):
            x = method.states(expiry_times[k-1])
            H = np.zeros(x.shape[-1])
        else:
            (x, H) = method.roll_back(expiry_times[k-1],expiry_times[k],x,U,H)
        U = underlying_payoffs[k-1].at(x.reshape((-1,x.shape[-1])))
    (x, H) = method.roll_back(0.0,expiry_times[0],x,U,H)
    if len(x.shape)>1:  # MC simulation
        x = x[0]
    npv = np.interp(0.0, x, H)
    return npv
