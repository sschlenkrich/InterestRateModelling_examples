
from faulthandler import disable
import sys
sys.path.append('./')

import numpy as np
import unittest
from tqdm import tqdm

from src.hull_white_model import HullWhiteModel
from src.methods.amc_solver import AmcSolver
from src.methods.amc_solver import AmcSolverOnlyExerciseRegression
from src.methods.amc_solver import CoterminalRateControls
from src.methods.density_integrations import DensityIntegrationWithBreakEven
from src.methods.density_integrations import CubicSplineExactIntegration
from src.methods.density_integrations import HermiteIntegration
from src.methods.density_integrations import SimpsonIntegration
from src.methods.pde_solver import PdeSolver
from src.monte_carlo_simulation import MonteCarloSimulation
from src.yieldcurve import FlatForwardCurve


class TestPricingMethods(unittest.TestCase):
    """
    Test density integration methods.
    """

    def test_martingale_property(self):
        curve = FlatForwardCurve(0.03)
        mean_reversion = 0.05
        vol_times = np.array([0.0])
        vol_vals  = np.array([0.01])
        model = HullWhiteModel(curve, mean_reversion, vol_times, vol_vals)
        # we need a MC simulation for AMC method
        times = np.linspace(0.0, 10.0, 11)
        n_paths = 2**16
        sim = MonteCarloSimulation(model, times, n_paths, showProgress=True)
        #
        T0 = 5.0
        T1 = 10.0
        T2 = 20.0
        #
        methods = [
            CubicSplineExactIntegration(model),
            HermiteIntegration(model, 5),
            SimpsonIntegration(model),
            #
            DensityIntegrationWithBreakEven(CubicSplineExactIntegration(model)),
            DensityIntegrationWithBreakEven(HermiteIntegration(model, 5)),
            DensityIntegrationWithBreakEven(SimpsonIntegration(model)),
            #
            PdeSolver(model),
            #
            AmcSolver(sim, 2),
            AmcSolverOnlyExerciseRegression(sim, 2),
            AmcSolver(sim, 2, controls=CoterminalRateControls(model, T2)),
            AmcSolverOnlyExerciseRegression(sim, 2, controls=CoterminalRateControls(model, T2)),
            AmcSolver(sim, 1, controls=CoterminalRateControls(model, T2, strike_rate=0.0)),
        ]
        ref_errors = (
            # min_T0                  median_T0               max_T0                  err_0
            ( 2.9791868971784651e-11, 3.9457371468718898e-07, 6.1914217781701517e-04, 8.9946598041956707e-08, ),
            ( 4.7388981627705104e-12, 1.5156328044459619e-11, 2.0873646811025994e-09, 2.4138230303449860e-11, ),
            ( 1.5636666036063693e-11, 3.9473955609561904e-07, 6.1915668632156104e-04, 8.9947819892355335e-08, ),
            ( 2.9791868971784651e-11, 3.9457371468718898e-07, 6.1914217781701517e-04, 8.9946598041956707e-08, ),
            ( 4.7388981627705104e-12, 1.5156328044459619e-11, 2.0873646811025994e-09, 2.4138230303449860e-11, ),
            ( 1.5636666036063693e-11, 3.9473955609561904e-07, 6.1915668632156104e-04, 8.9947819892355335e-08, ),
            ( 1.0885062251553278e-08, 7.9397105917922146e-07, 9.9617685833234648e-04, 1.4503581369229579e-07, ),
            ( 7.5891530097275484e-11, 2.0799355032411667e-04, 1.5434289119059041e-02, 2.0168759827127182e-05, ),
            ( 2.0111106640019473e-07, 9.4620512875760870e-03, 5.8127638102258530e-02, 1.0598903143438364e-05, ),
            ( 7.5891607812887215e-11, 2.0799355032412960e-04, 1.5434289119058805e-02, 2.0168759827188243e-05, ),
            ( 2.0111106640019473e-07, 9.4620512875760870e-03, 5.8127638102258530e-02, 1.0598903143438364e-05, ),
            ( 1.4835285141086748e-08, 1.1517828637251056e-03, 8.0823181086123352e-02, 2.0116491925364909e-05, ),
        )
        print('')
        for method, ref_err in tqdm(zip(methods, ref_errors), 'Method', disable=False):
            x1 = method.states(T1)
            U1 = model.zero_bond(T1, x1 if len(x1.shape)==1 else x1[0], T2)
            H1 = np.zeros(U1.shape)
            x0, H0 = method.roll_back(T0, T1, x1, U1, H1)
            U0 = model.zero_bond(T0, x0 if len(x0.shape)==1 else x0[0], T2)
            error_T0 = np.abs(np.log(H0) - np.log(U0))/(T2-T0)  # as spread rate
            #
            x0, H0 = method.roll_back(0.0, T0, x0, H0, H1)
            P_0_T2 = np.interp(0.0, x0 if len(x0.shape)==1 else x0[0], H0)
            error_0 = np.abs(np.log(P_0_T2) - np.log(curve.discount(T2)))/(T2-0.0)
            #
            # print('( %.16e, %.16e, %.16e, %.16e, ),' % 
            #     (np.min(error_T0),
            #      np.median(error_T0),
            #      np.max(error_T0),
            #      error_0,
            #     ))
            self.assertLessEqual(np.min(error_T0),    ref_err[0])
            self.assertLessEqual(np.median(error_T0), ref_err[1])
            self.assertLessEqual(np.max(error_T0),    ref_err[2])
            self.assertLessEqual(error_0,             ref_err[3])


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestPricingMethods))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())

