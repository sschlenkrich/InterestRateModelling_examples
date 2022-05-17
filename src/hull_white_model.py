
import numpy as np
from scipy.optimize import brentq

from src.helpers import black


class HullWhiteModel:
    """
    Hull-White interest rate model with analytic pricing formulas
    and stochastic process interface.
    """

    # Python constructor
    def __init__(self, yield_curve, mean_reversion, volatility_times, volatility_values):
        self.yield_curve       = yield_curve
        self.mean_reversion    = mean_reversion
        self.volatility_times  = volatility_times    # assume positive and ascending
        self.volatility_values = volatility_values
        # pre-calculate y(t) on the time grid
        # y(t) = G'(s,t)^2 y(s) + sigma^2 [1 - exp{-2a(t-s)}] / (2a)
        t0 = 0.0
        y0 = 0.0
        self.y_ = np.zeros(len(self.volatility_times))
        for i in range(len(self.y_)):
            self.y_[i] = (self.G_prime(t0,self.volatility_times[i])**2) * y0 +                   \
                         (self.volatility_values[i]**2) *                                       \
                         (1.0 - np.exp(-2*self.mean_reversion*(self.volatility_times[i]-t0))) /  \
                         (2.0 * self.mean_reversion)
            t0 = self.volatility_times[i]
            y0 = self.y_[i]
        
    def sigma(self,t):
        """Short rate volatility."""
        idx = np.searchsorted(self.volatility_times,t)
        return self.volatility_values[min(idx,len(self.volatility_values)-1)]

    def G(self, t, T):
        return (1.0 - np.exp(-self.mean_reversion*(T-t))) / self.mean_reversion
    
    def G_prime(self, t, T):
        return np.exp(-self.mean_reversion*(T-t))
        
    def y(self,t):
        """Auxilliary state variable."""
        # find idx s.t. t[idx-1] < t <= t[idx]
        idx = np.searchsorted(self.volatility_times,t)
        t0 = 0.0 if idx==0 else self.volatility_times[idx-1]
        y0 = 0.0 if idx==0 else self.y_[idx-1]
        s1 = self.volatility_values[min(idx,len(self.volatility_values)-1)]  # flat extrapolation
        y1 = (self.G_prime(t0,t)**2) * y0 +                      \
                s1**2 * (1.0 - np.exp(-2*self.mean_reversion*(t-t0))) /  \
                (2.0 * self.mean_reversion)
        return y1

    def risk_neutral_expectation(self, t, xt, T):
        """Conditional expectation in risk-neutral measure."""
        # E[x] = G'(t,T)x + \int_t^T G'(u,T)y(u)du
        def f(u):
            return self.G_prime(u,T)*self.y(u)
        # use Simpson's rule to approximate integral, this should better be solved analytically
        integral = (T-t) / 6 * (f(t) + 4*f((t+T)/2) + f(T)) 
        return self.G_prime(t,T)*xt + integral
    
    def T_forward_expectation(self, t, xt, T):
        """Conditional expectation in T-forward measure."""
        return self.G_prime(t,T)*(xt + self.G(t,T)*self.y(t))

    def variance(self, t, T):
        """Conditional variance."""
        return self.y(T) - self.G_prime(t,T)**2 * self.y(t)

    def zero_bond(self, t, xt, T):
        """Zero coupon bond formula."""
        G = self.G(t,T)
        return self.yield_curve.discount(T) / self.yield_curve.discount(t) * \
            np.exp(-G*xt - 0.5 * G**2 * self.y(t) )

    def zero_bond_option(self, expiry_time, maturityTime, strikePrice, call_or_put):
        """Zero coupon bond option formula."""
        nu2 = self.G(expiry_time,maturityTime)**2 * self.y(expiry_time)
        P0  = self.yield_curve.discount(expiry_time)
        P1  = self.yield_curve.discount(maturityTime)
        return P0 * black(strikePrice,P1/P0,np.sqrt(nu2),1.0, call_or_put)

    def coupon_bond_option(self, expiry_time, pay_times, cash_flows, strike_price, call_or_put):
        """Coupon bond option formula using Jamschidian's trick."""
        def objective(x):
            bond = 0
            for i in range(len(pay_times)):
                bond += cash_flows[i] * self.zero_bond(expiry_time,x,pay_times[i])
            return bond - strike_price
        x_star = brentq(objective,-1.0, 1.0, xtol=1.0e-8)
        bondOption = 0.0
        for i in range(len(pay_times)):
            strike = self.zero_bond(expiry_time, x_star, pay_times[i])
            bondOption += cash_flows[i] * self.zero_bond_option(expiry_time, pay_times[i], strike, call_or_put)
        return bondOption

    # future yield curve in terms of forward rates
    def forward_rate(self, t, xt, T):
        """Future yield curve in terms of forward rates."""
        return self.yield_curve.forwardRate(T) + self.G_prime(t,T)*(xt + self.G(t,T)*self.y(t))

    # stochastic process interface
    
    def size(self):   # dimension of X(t)
        return 2      # [x, s], we also need for numeraire s = \int_0^t r dt

    def factors(self):   # dimension of W(t)
        return 1

    def initial_values(self):
        return np.array([ 0.0, 0.0 ])
    
    def zero_bond_payoff(self, X, t, T):
        """Zero bond price for a given simulated state X."""
        return self.zero_bond(t, X[0], T)

    def numeraire(self, X, t):
        """Numeraire price for a given simulated state X."""
        return np.exp(X[1]) / self.yield_curve.discount(t)

    def evolve(self, t0, X0, dt, dW):
        """
        Evolve X(t0) -> X(t0+dt) using independent Brownian increments dW.
        Inputs t0, dt are assumed float, X0, X1, dW are np.array.
        """
        x1 = self.risk_neutral_expectation(t0,X0[0],t0+dt)
        # x1 = X0[0] + (self.y(t0) - self.mean_reversion*X0[0])*dt
        nu = np.sqrt(self.variance(t0,t0+dt))
        x1 = x1 + nu*dW[0]
        # s1 = s0 + \int_t0^t0+dt x dt via Trapezoidal rule
        s1 = X0[1] + (X0[0] + x1) * dt / 2
        # gather results
        return np.array([x1, s1])          
        

class HullWhiteModelWithDiscreteNumeraire(HullWhiteModel):
    """
    Hull White model which is simulated in rolling T-forward
    measure.
    """

    # Python constructor
    def __init__(self, yield_curve, mean_reversion, volatility_times, volatility_values):
        HullWhiteModel.__init__(self,yield_curve, mean_reversion, volatility_times, volatility_values)

    def evolve(self, t0, X0, dt, dW):
        """
        Evolve X(t0) -> X(t0+dt) using independent Brownian increments dW.
        Inputs t0, dt are assumed float, X0, X1, dW are np.array.
        Simulation is done with discretely compounded bank account numeraire
        and rolling T-forward measure.
        """
        x1 = self.T_forward_expectation(t0,X0[0],t0+dt)
        nu = np.sqrt(self.variance(t0,t0+dt))
        x1 = x1 + nu*dW[0]
        s1 = X0[1] + np.log(1.0/self.zero_bond(t0,X0[0],t0+dt))
        return np.array([x1, s1])          
