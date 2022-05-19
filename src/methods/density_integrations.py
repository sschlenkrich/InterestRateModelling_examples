
import numpy as np
from scipy.stats import norm
from scipy import integrate
from scipy.interpolate import CubicSpline


class DensityIntegration:  # base class for other integration methods
    """
    A base class for various integration methods.
    """

    # Python constructor
    def __init__(self, hwModel, nGridPoints=101, stdDevs=5):
        self.hwModel     = hwModel
        self.nGridPoints = nGridPoints
        self.stdDevs     = stdDevs
    
    def states(self, expityTime):
        sigma = np.sqrt(self.hwModel.variance(0.0, expityTime))
        if sigma==0:
            return np.zeros((1,))
        return np.linspace(-self.stdDevs*sigma,self.stdDevs*sigma,self.nGridPoints)

    def roll_back(self, T0, T1, x1, U1, H1):
        """Method needs to be implemented in derived classes."""
        raise NotImplementedError

class DensityIntegrationWithBreakEven(DensityIntegration):
    """
    Decorate a method with break-even methodology.
    """

    # Python constructor
    def __init__(self, method):
        DensityIntegration.__init__(self,method.hwModel,method.nGridPoints,method.stdDevs)
        self.method = method
    
    def roll_back(self, T0, T1, x1, U1, H1):
        # find break-even state and split grid
        roots = CubicSpline(x1,U1-H1).roots(discontinuity=False, extrapolate=False)
        if roots.shape[0]==0:  # no break even point found
            return self.method.roll_back(T0,T1,x1,U1,H1)
        xStar = roots[0] 
        VStar = CubicSpline(x1,U1)(xStar)
        # lower integrand
        lRange = np.where(x1 < xStar)
        lX1 = np.concatenate((x1[lRange], [xStar]))
        lU1 = np.concatenate((U1[lRange], [VStar]))
        lH1 = np.concatenate((H1[lRange], [VStar]))
        #
        (x0, lV0) = self.method.roll_back(T0,T1,lX1,lU1,lH1)
        # upper integrand
        uRange = np.where(x1 > xStar)
        uX1 = np.concatenate(([xStar], x1[uRange]))
        uU1 = np.concatenate(([VStar], U1[uRange]))
        uH1 = np.concatenate(([VStar], H1[uRange]))
        # 
        (x0, uV0) = self.method.roll_back(T0,T1,uX1,uU1,uH1)
        # combine integrations
        return (x0, lV0+uV0)


class SimpsonIntegration(DensityIntegration):

    # Python constructor
    def __init__(self, hwModel, nGridPoints=101, stdDevs=5):
        DensityIntegration.__init__(self,hwModel,nGridPoints,stdDevs)
    
    def roll_back(self, T0, T1, x1, U1, H1):
        x0 = self.states(T0)
        V0 = np.zeros(x0.shape[0])
        sigma = np.sqrt(self.hwModel.variance(T0,T1))
        V = np.maximum(U1, H1)
        for i in range(x0.shape[0]):
            mu = self.hwModel.T_forward_expectation(T0, x0[i], T1)
            fx = np.array([ V[k] * norm.pdf((x1[k]-mu)/sigma)/sigma for k in range(x1.shape[0])])
            I = integrate.simps(fx, x1)
            V0[i] = self.hwModel.zero_bond(T0,x0[i],T1) * I
        return (x0, V0)

    
class HermiteIntegration(DensityIntegration):

    # Python constructor
    def __init__(self, hwModel, degree, nGridPoints=101, stdDevs=5):
        DensityIntegration.__init__(self,hwModel,nGridPoints,stdDevs)
        (self.hermX, self.hermW) = np.polynomial.hermite.hermgauss(degree)

    def roll_back(self, T0, T1, x1, U1, H1):
        x0 = self.states(T0)
        V0 = np.zeros(x0.shape[0])
        sigma = np.sqrt(self.hwModel.variance(T0,T1))
        V = CubicSpline(x1, np.maximum(U1, H1))
        for i in range(x0.shape[0]):
            mu = self.hwModel.T_forward_expectation(T0, x0[i], T1)
            I = 0.0
            for k in range(self.hermX.shape[0]):
                I += self.hermW[k] * V(np.sqrt(2.0)*sigma*self.hermX[k] + mu)
            I /= np.sqrt(np.pi)
            V0[i] = self.hwModel.zero_bond(T0,x0[i],T1) * I
        return (x0, V0)


class CubicSplineExactIntegration(DensityIntegration):

    # Python constructor
    def __init__(self, hwModel, nGridPoints=101, stdDevs=5):
        DensityIntegration.__init__(self,hwModel,nGridPoints,stdDevs)

    def roll_back(self, T0, T1, x1, U1, H1):
        x0 = self.states(T0)
        V0 = np.zeros(x0.shape[0])
        sigma = np.sqrt(self.hwModel.variance(T0,T1))
        V = CubicSpline(x1, np.maximum(U1, H1))
        for i in range(x0.shape[0]):
            mu = self.hwModel.T_forward_expectation(T0, x0[i], T1)
            # we need to setup all the coefficients
            xBar     = (V.x - mu)/sigma
            Phi      = norm.cdf(xBar)
            PhiPrime = norm.pdf(xBar)
            F0 = Phi
            F1 = -1.0*PhiPrime
            F2 = Phi - xBar*PhiPrime
            F3 = -1.0*(xBar**2 + 2.0)*PhiPrime
            dF0 = F0[1:] - F0[:-1]
            dF1 = F1[1:] - F1[:-1]
            dF2 = F2[1:] - F2[:-1]
            dF3 = F3[1:] - F3[:-1]
            I0  = dF0
            I1  = sigma*dF1 - sigma*xBar[:-1]*I0
            I2  = (sigma**2)*dF2 - 2*sigma*xBar[:-1]*I1 - (sigma**2)*(xBar[:-1]**2)*I0
            I3  = (sigma**3)*dF3 - 3*sigma*xBar[:-1]*I2 - 3*(sigma**2)*(xBar[:-1]**2)*I1 - (sigma**3)*(xBar[:-1]**3)*I0
            # summing up
            I = 0.0
            for k in range(xBar.shape[0]-1):
                I += V.c[3][k]*I0[k] + V.c[2][k]*I1[k] + V.c[1][k]*I2[k] + V.c[0][k]*I3[k]
            V0[i] = self.hwModel.zero_bond(T0,x0[i],T1) * I
        return (x0, V0)