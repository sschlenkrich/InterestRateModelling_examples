
import sys
sys.path.append('./')

import matplotlib.pyplot as plt
import numpy as np
import unittest

from src.hull_white_model import HullWhiteModel

class FlatForwardCurve:
    """A helper class for yield curve."""
    def __init__(self, forward_rate):
        self.__forward_rate__ = forward_rate
        self.discount         = lambda T : np.exp(-self.__forward_rate__*T)
        self.forwardRate      = lambda T : self.__forward_rate__
    

class TestHullWhiteModel(unittest.TestCase):
    """
    Test Hull-White model construction and option pricing.
    """

    def test_hull_white_model_setup_and_vol_interpolation(self):
        discount_curve    = FlatForwardCurve(0.02)
        mean_reversion    = 0.03
        volatility_times  = np.array([ 1.0, 2.0, 5.0 ])
        volatility_values = np.array([ 100, 100, 100 ]) * 1e-4
        model = HullWhiteModel(discount_curve, mean_reversion, volatility_times, volatility_values)
        model_flat = HullWhiteModel(discount_curve, mean_reversion, 
            volatility_times  = np.array([ 1.0 ]), 
            volatility_values = np.array([ 100 ]) * 1e-4,
        )
        test_times = np.array([ 0.0, 0.5, 1.0, 1.5, 2.0, 3.5, 5.0, 6.0 ])
        for t in test_times:
            self.assertEqual(model.sigma(t), model_flat.sigma(t))
            self.assertLess(np.abs(model.y(t) - model_flat.y(t)), 1.0e-16)

    def test_hull_white_analytic_formulas(self):
        discount_curve    = FlatForwardCurve(0.02)
        mean_reversion    = 0.03
        volatility_times  = np.array([ 1.0, 2.0, 5.0 ])
        volatility_values = np.array([ 100,  80,  70 ]) * 1e-4
        model = HullWhiteModel(discount_curve, mean_reversion, volatility_times, volatility_values)
        # we test against reference values
        zcb = model.zero_bond(10.0, 0.01, 20.0)
        # print(zcb)
        self.assertLess(np.abs(zcb - 0.739665065962519), 1.0e-16)
        zb_call = model.zero_bond_option(10.0, 20.0, 1.0, 1.0)
        # print(zb_call)
        self.assertLess(np.abs(zb_call - 0.008017065488223098), 1.0e-16)
        zb_put = model.zero_bond_option(10.0, 20.0, 1.0, -1.0)
        # print(zb_put)
        self.assertLess(np.abs(zb_put - 0.1564277725305656), 1.0e-16)
        # put/call parity
        self.assertLess(np.abs(
            (zb_call - zb_put)/discount_curve.discount(10.0) - 
            (discount_curve.discount(20.0)/discount_curve.discount(10.0) - 1.0)
            ),
            1.0e-16)
        #
        expiry_time = 10.0
        pay_times = np.array([ 11.0, 12.0, 13.0, 14.0, 15.0, 15.0 ])
        cash_flows = np.array([ 0.02, 0.02, 0.02, 0.02, 0.02, 1.00 ])
        strike_price = 1.0
        call_or_put = 1
        cb_call = model.coupon_bond_option(expiry_time, pay_times, cash_flows, strike_price, call_or_put)
        # print(cb_call)
        self.assertLess(np.abs(cb_call - 0.029017570861380355), 1.0e-16)
        #
        pay_times  = np.concatenate(([10.0], pay_times))
        cash_flows = np.concatenate(([-1.0], cash_flows))
        cb_call_zero_strike = model.coupon_bond_option(expiry_time, pay_times, cash_flows, 0.0, call_or_put)
        self.assertLess(np.abs(cb_call - cb_call_zero_strike), 1.0e-16)
        #
        pay_times  = np.array([ 20.0 ])
        cash_flows = np.array([ 1.00 ])
        cb_call_single = model.coupon_bond_option(expiry_time, pay_times, cash_flows, strike_price, call_or_put)
        self.assertLess(np.abs(cb_call_single - zb_call), 2.0e-12)

    def test_hull_white_forward_rate(self):
        discount_curve    = FlatForwardCurve(0.02)
        mean_reversion    = 0.03
        volatility_times  = np.array([ 1.0, 2.0, 5.0 ])
        volatility_values = np.array([ 100,  80,  70 ]) * 1e-4
        model = HullWhiteModel(discount_curve, mean_reversion, volatility_times, volatility_values)
        #
        t = 5.0
        dT = np.linspace(0.0, 10.0, 11)
        xt = np.linspace(-0.10, 0.10, 11)
        plt.figure()
        for x in reversed(xt):
            f = np.array([ model.forward_rate(t, x, t+delta) for delta in dT])
            plt.plot(t+dT, f, label='$x_t=%.3f$' % x)
        plt.legend()
        plt.xlim((0.0, t+dT[-1]))
        plt.xlabel('time $T$')
        plt.ylabel('forward rate $f(t,T)$')
        # plt.show()


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestHullWhiteModel))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())

