
import numpy as np
from scipy.optimize import brentq

from src.helpers import bachelier

class SabrModel:

    # Python constructor
    def __init__(self, forward, time_to_expiry, alpha, beta, nu, rho):
        self.forward      = forward
        self.time_to_expiry = time_to_expiry
        self.alpha        = alpha
        self.beta         = beta
        self.nu           = nu
        self.rho          = rho

    # helpers
    def local_vol_C(self, rate):
        rate = np.maximum(rate, 1.0e-7) # avoid errors in power
        return np.power(rate,self.beta)
        
    def local_vol_C_prime(self, rate):  # for Milstein method
        rate = np.maximum(rate, 1.0e-7) # avoid errors in power
        return self.beta * np.power(rate,self.beta-1)
        
    def s_average(self, strike, forward):
        return (strike + forward) / 2.0
    
    def zeta(self, strike, forward):
        return self.nu / self.alpha * \
            (np.power(forward,1-self.beta)-np.power(strike,1-self.beta)) / \
            (1-self.beta)
        
    def chi(self, zeta):
        return np.log((np.sqrt(1-2*self.rho*zeta+zeta**2)-self.rho+zeta)/(1-self.rho))
     
    def normal_volatility(self,strike):
        """Approximate implied normal volatility formula."""
        Sav     = self.s_average(strike,self.forward)
        CSav    = self.local_vol_C(Sav)
        gamma1  = self.beta / Sav
        gamma2  = self.beta * (self.beta-1) / Sav**2
        I1      = (2*gamma2 - gamma1**2) / 24 * self.alpha**2 * CSav**2
        I1      = I1 + self.rho * self.nu * self.alpha * gamma1 / 4 * CSav
        I1      = I1 + (2 - 3*self.rho**2) / 24 * self.nu**2
        sigmaN  = self.alpha * CSav  # default, if close to ATM
        atm_eps = 1.0e-8
        if np.fabs(strike-self.forward)>atm_eps:  # actual calculation for I0
            sigmaN = self.nu * (self.forward - strike) / self.chi(self.zeta(strike,self.forward))
        sigmaN  = sigmaN * (1 + I1*self.time_to_expiry)  # higher order adjustment
        return sigmaN

    def calibrate_atm(self, sigma_atm):
        """Calibrate alpha s.t. model matches given at-the-money vol"""
        def objective(alpha):
            self.alpha = alpha
            return self.normal_volatility(self.forward) - sigma_atm
        alpha0 = sigma_atm / self.local_vol_C(self.forward)
        self.alpha = brentq(objective,0.5*alpha0, 2.0*alpha0, xtol=1.0e-8)
        return self.alpha

    def vanilla_price(self, strike, call_or_put):
        sigmaN = self.normal_volatility(strike)
        return bachelier(strike,self.forward,sigmaN,self.time_to_expiry,call_or_put)

    def density(self, rate):
        eps = 1.0e-4
        cop = 1.0
        if (rate<self.forward):
            cop = -1.0
        dens = (self.vanilla_price(rate-eps,cop) - 2*self.vanilla_price(rate,cop) + self.vanilla_price(rate+eps,cop))/eps/eps
        return dens

    # stochastic process interface
    
    def size(self):   # dimension of X(t)
        return 2

    def factors(self):   # dimension of W(t)
        return 2

    def initial_values(self):
        return np.array([ self.forward, self.alpha ])
    
    def evolve(self, t0, X0, dt, dW):
        """
        Evolve X(t0) -> X(t0+dt) using independent Brownian increments dW.

        We implement a vectorised time stepping.

        t0, dt are assumed float,
        X0, dW are 2d array of shape (2, n_paths).

        Returns X1 as 2d array of shape (2, n_paths).
        """
        # first simulate stochastic volatility exact
        dZ = self.rho * dW[0] + np.sqrt(1-self.rho*self.rho)*dW[1]
        alpha0 = X0[1]
        alpha1 = alpha0*np.exp(-self.nu*self.nu/2*dt+self.nu*dZ*np.sqrt(dt))
        alpha01 = np.sqrt(alpha0*alpha1)   # average vol [t0, t0+dt]
        # simulate S via Milstein
        S0 = X0[0]
        S1 = S0 + alpha01*self.local_vol_C(S0)*dW[0]*np.sqrt(dt) \
                + 0.5*alpha01*self.local_vol_C(S0)*alpha01*self.local_vol_C_prime(S0)*(dW[0]*dW[0]-1)*dt 
        # gather results
        return np.array([S1, alpha1])          
