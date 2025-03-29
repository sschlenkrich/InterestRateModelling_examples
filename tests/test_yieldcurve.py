import sys
sys.path.append('./')

import numpy as np
import QuantLib as ql
import unittest

from src.yieldcurve import YieldCurve

class TestYieldCurve(unittest.TestCase):
    """
    Test yield curve construction and discount factor calculation.
    """

    def test_versus_quantlib(self):
        #
        terms = [ '1y' ]
        rates = [ 0.03 ]
        yc = YieldCurve(terms, rates)
        #
        today = ql.Settings.instance().evaluationDate
        ql_ytsh = ql.YieldTermStructureHandle(
            ql.FlatForward(today,rates[0],ql.Actual365Fixed()))
        #
        maturity_time = 1.234
        #
        self.assertEqual(yc.discount(maturity_time), np.exp(-rates[0]*maturity_time))
        self.assertEqual(yc.discount(maturity_time), ql_ytsh.discount(maturity_time))
        self.assertEqual(yc.discount(today + 42), ql_ytsh.discount(today + 42))
        #
        self.assertAlmostEqual(yc.forwardRate(maturity_time), rates[0], places=11)
        self.assertAlmostEqual(
            yc.forwardRate(maturity_time),
            ql_ytsh.forwardRate(maturity_time, maturity_time, ql.Continuous).rate(),
            places=11)
        return None

    def test_table(self):
        terms = [ '1y', '5y', '10y' ]
        rates = [ 0.01, 0.02, 0.03  ]
        yc = YieldCurve(terms, rates)
        print('')
        print(yc.table())


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestYieldCurve))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())

