import sys
sys.path.append('./')

import numpy as np
import QuantLib as ql
import unittest

from src.swap import Swap
from src.yieldcurve import YieldCurve

class TestSwap(unittest.TestCase):
    """
    Test Vanilla swap construction and pricing.
    """

    def test_swap_setup(self):
        today = ql.Date(3,9,2018)
        ql.Settings.instance().evaluationDate = today
        discCurve = YieldCurve(['30y'], [0.03])
        projCurve = YieldCurve(['30y'], [0.04])
        startDate = ql.Date(30, 10, 2018)
        endDate = ql.Date(30, 10, 2038)
        swap = Swap(startDate,endDate,0.05,discCurve,projCurve)
        #
        print('')
        print('NPV:      %11.2f' % (swap.npv()))
        print('FairRate: %11.6f' % (swap.fairRate()))
        print('Annuity:  %11.2f' % (swap.annuity()))
        #
        print(swap.fixedCashFlows())
        print(swap.floatCashFlows())


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestSwap))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())

