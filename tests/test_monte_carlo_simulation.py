import sys
sys.path.append('./')

import numpy as np
import matplotlib.pyplot as plt
import unittest

from src.helpers import bachelier_implied_vol
from src.monte_carlo_simulation import MonteCarloSimulation
from src.sabr_model import SabrModel
from src.hull_white_model import HullWhiteModel

def implied_volatility(sim, T, strikes):
    assert sim.times[-1] == T
    S_0 = sim.X[0,0,0]
    S_T = sim.X[-1,0,:]
    S_T += (S_0 - S_T.mean()) # we incorporate an adjuster to numerically ensure put-call-parity
    S_T = np.reshape(S_T, (-1,1))
    K = np.reshape(strikes, (1,-1))
    #
    V_T = np.maximum((2*(K>S_0)-1) * (S_T - K), 0.0)
    E_T_T = np.mean(V_T, axis=0)
    vols = np.array([
         bachelier_implied_vol(P_, K_, S_0, T, 2*(K_>S_0)-1) 
         for P_, K_ in zip(E_T_T, strikes)
         ])
    return vols

class FlatForwardCurve:
    """A helper class for yield curve."""
    def __init__(self, forward_rate):
        self.__forward_rate__ = forward_rate
        self.discount         = lambda T : np.exp(-self.__forward_rate__*T)
        self.forwardRate      = lambda T : self.__forward_rate__


class TestMonteCarloSimulation(unittest.TestCase):
    """
    Test Monte Carlo simulation with SABR and and Hull-White model.
    """

    def test_mc_with_sabr_model(self):
        S0 = 0.05 # forward rate 5%
        T = 5.0 # 5y expiry
        # SabrModel( S(t), T, alpha, beta, nu, rho )
        model1 = SabrModel(S0, T, 0.0100, 0.0001, 0.0001, 0.0)
        model2 = SabrModel(S0, T, 0.0450, 0.5000, 0.0001, 0.0)
        model3 = SabrModel(S0, T, 0.0405, 0.5000, 0.5000, 0.0)
        model4 = SabrModel(S0, T, 0.0420, 0.5000, 0.5000, 0.7)
        #
        print('')
        times = np.linspace(0.0, T, 501)
        n_paths = 2**13
        sim1 = MonteCarloSimulation(model1,times,n_paths, showProgress=True)
        sim2 = MonteCarloSimulation(model2,times,n_paths, showProgress=True)
        sim3 = MonteCarloSimulation(model3,times,n_paths, showProgress=True)
        sim4 = MonteCarloSimulation(model4,times,n_paths, showProgress=True)
        #
        print(sim1.X.shape)
        #
        ref_strikes = np.linspace(0.01, 0.10, 10)
        vols1 = implied_volatility(sim1, T, ref_strikes)
        vols2 = implied_volatility(sim2, T, ref_strikes)
        vols3 = implied_volatility(sim3, T, ref_strikes)
        vols4 = implied_volatility(sim4, T, ref_strikes)
        plt.plot(ref_strikes, vols1)
        plt.plot(ref_strikes, vols2)
        plt.plot(ref_strikes, vols3)
        plt.plot(ref_strikes, vols4)
        # plt.show()


    def test_mc_with_hull_white_model(self):
        discount_curve    = FlatForwardCurve(0.02)
        mean_reversion    = 0.03
        volatility_times  = np.array([ 1.0, 2.0, 5.0 ])
        volatility_values = np.array([ 100,  80,  70 ]) * 1e-4
        model = HullWhiteModel(discount_curve, mean_reversion, volatility_times, volatility_values)
        print('')
        times = np.linspace(0.0, 10.0, 11)
        n_paths = 2**16
        sim = MonteCarloSimulation(model,times,n_paths, showProgress=True)
        self.assertEqual(sim.X.shape, (len(times), model.size(), n_paths))
        zcb = model.zero_bond_payoff(sim.X[-1,:,:], 10.0, 20.0)
        num = model.numeraire(sim.X[-1,:,:], 10.0)
        zcb_0T = np.mean(zcb / num)
        mc_spread = np.log(zcb_0T / discount_curve.discount(20.0)) / 20.0
        # print(mc_spread)
        self.assertLess(np.abs(mc_spread), 5.0e-5)



if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestMonteCarloSimulation))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())

