
from scipy.stats import norm
from scipy.optimize import brentq
import numpy as np


def black_normalised(moneyness, stdDev, callOrPut):
    d1 = np.log(moneyness) / stdDev + stdDev / 2.0
    d2 = d1 - stdDev
    return callOrPut * (moneyness*norm.cdf(callOrPut*d1)-norm.cdf(callOrPut*d2))

def black(strike, forward, sigma, T, callOrPut):
    nu = sigma*np.sqrt(T)
    if nu<1.0e-12:   # assume zero
        return max(callOrPut*(forward-strike),0.0)  # intrinsic value
    return strike * black_normalised(forward/strike,nu,callOrPut)

def black_implied_vol(price, strike, forward, T, callOrPut):
    def objective(sigma):
        return black(strike, forward, sigma, T, callOrPut) - price
    return brentq(objective,0.01, 1.00, xtol=1.0e-8)

def bachelier_normalised(moneyness, stdDev, callOrPut):
    h = callOrPut * moneyness / stdDev
    return stdDev * (h*norm.cdf(h) + norm.pdf(h))

def bachelier_vega_normalised(moneyness, stdDev):
    return norm.pdf(moneyness / stdDev)

def bachelier(strike, forward, sigma, T, callOrPut):
    return bachelier_normalised(forward-strike,sigma*np.sqrt(T),callOrPut)

def bachelier_vega(strike, forward, sigma, T):
    return bachelier_vega_normalised(forward-strike,sigma*np.sqrt(T))*np.sqrt(T)

def bachelier_implied_vol(price, strike, forward, T, callOrPut):
    def objective(sigma):
        return bachelier(strike, forward, sigma, T, callOrPut) - price
    return brentq(objective,1e-4, 1e-1, xtol=1.0e-8)

