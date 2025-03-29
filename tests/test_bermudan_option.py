
import sys
sys.path.append('./')

import numpy as np
import unittest

from src.bermudan_option import bermudan_option_npv
from src.hull_white_model import HullWhiteModel
from src.methods.amc_solver import AmcSolver
from src.methods.amc_solver import AmcSolverOnlyExerciseRegression
from src.methods.amc_solver import CoterminalRateControls
from src.methods.density_integrations import DensityIntegrationWithBreakEven
from src.methods.density_integrations import CubicSplineExactIntegration
from src.methods.density_integrations import HermiteIntegration
from src.methods.density_integrations import SimpsonIntegration
from src.methods.payoffs import CouponBond
from src.methods.pde_solver import PdeSolver
from src.monte_carlo_simulation import MonteCarloSimulation
from src.yieldcurve import FlatForwardCurve


class TestBermudanOption(unittest.TestCase):
    """
    Test Bermudan option backward induction algorithm.
    """

    def test_coupon_bond(self):
        curve = FlatForwardCurve(0.03)
        mean_reversion = 0.05
        vol_times = np.array([0.0])
        vol_vals  = np.array([0.01])
        model = HullWhiteModel(curve, mean_reversion, vol_times, vol_vals)
        # we need a MC simulation for AMC method
        times = np.linspace(0.0, 20.0, 21)
        n_paths = 2**16
        sim = MonteCarloSimulation(model, times, n_paths, showProgress=True)
        #
        exercise  = 12.0
        payTimes  = [ 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 20.0 ]
        cashFlows = [ -1.0, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,  1.0 ]
        expiryTimes = np.array([ 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
        underlyings = []
        for k in range(8):
            pTimes = [payTimes[k]] + payTimes[(k+1):]
            cFlows = [-1.0 ] + cashFlows[(k+1):]
            underlyings.append(CouponBond(model,payTimes[k],pTimes,cFlows))
        #
        methods = [
            CubicSplineExactIntegration(model),
            HermiteIntegration(model, 5),
            SimpsonIntegration(model),
            #
            DensityIntegrationWithBreakEven(CubicSplineExactIntegration(model)),
            DensityIntegrationWithBreakEven(SimpsonIntegration(model)),
            #
            PdeSolver(model),
            #
            AmcSolver(sim, 2),
            AmcSolverOnlyExerciseRegression(sim, 2),
            AmcSolver(sim, 2, controls=CoterminalRateControls(model, payTimes[-1])),
            AmcSolverOnlyExerciseRegression(sim, 2, controls=CoterminalRateControls(model, payTimes[-1])),
            AmcSolver(sim, 1, controls=CoterminalRateControls(model, payTimes[-1], strike_rate=0.0)),
        ]
        europeans_npv = []
        europeans_npv_ref = []
        for T, bond in zip(expiryTimes, underlyings):
            europeans_npv_ref.append(model.coupon_bond_option(T, bond.pay_times, bond.cash_flows, 0.0, 1.0))
            europeans_npv.append(
                [ bermudan_option_npv([T], [bond], method) for method in methods ]
            )
        europeans_npv = np.array(europeans_npv).T
        europeans_npv_ref = np.array([europeans_npv_ref])
        err_min = np.min(np.abs(europeans_npv - europeans_npv_ref), axis=1)
        err_max = np.max(np.abs(europeans_npv - europeans_npv_ref), axis=1)
        err_ref = (
            # min                    max
            (5.5990403988528947e-07, 2.8666036795431393e-05),
            (9.3174374042802495e-04, 4.3387365039538051e-03),
            (7.6855484937646268e-07, 6.2531766233032970e-05),
            (2.5613915780910157e-08, 2.5216098777092188e-07),
            (5.3589026141089013e-08, 5.0240220359626386e-07),
            (9.5372220783405265e-07, 3.1825161067110252e-05),
            (3.3084830873358895e-05, 1.9274304337404957e-04),
            (3.3084830873358895e-05, 1.9274304337404957e-04),
            (3.3084830873358895e-05, 1.9274304337404957e-04),
            (3.3084830873358895e-05, 1.9274304337404957e-04),
            (3.3084830873358895e-05, 1.9274304337404957e-04),
        )
        for min_, max_, ref in zip(err_min, err_max, err_ref):
            # print('(%.16e, %.16e),' % (min_, max_))
            self.assertLessEqual(min_, ref[0])
            self.assertLessEqual(max_, ref[1])
        berms = []
        for method in methods:
            berms.append(bermudan_option_npv(expiryTimes, underlyings, method, showProgress=True))
        npv_refs = (
            4.8431440607371520e-02,
            4.8592837407837541e-02,
            4.8425607015956156e-02,
            4.8435049056882451e-02,
            4.8435276198547690e-02,
            4.8430247218415087e-02,
            4.7918687011413386e-02,
            5.2557114160640774e-02,
            4.7918687011402977e-02,
            5.2557114160640774e-02,
            5.7489117656044170e-02,
        )
        for npv, npv_ref in zip(berms, npv_refs):
            # print('%.16e,' % npv)
            self.assertEqual(npv, npv_ref)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestBermudanOption))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())

