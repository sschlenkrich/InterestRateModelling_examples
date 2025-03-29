import sys
sys.path.append('./')


from pprint import pprint
import QuantLib as ql
import unittest

from src.swap import Swap
from src.swaption import Swaption
from src.yieldcurve import YieldCurve

class TestSwaption(unittest.TestCase):
    """
    Test European swaption construction and pricing.
    """

    def test_swaption_setup(self):
        today = ql.Date(3,9,2018)
        ql.Settings.setEvaluationDate(ql.Settings.instance(),today)
        discCurve = YieldCurve(['30y'], [0.03])
        projCurve = YieldCurve(['30y'], [0.04])
        startDate = ql.Date(3, 9, 2028)
        endDate = ql.Date(3, 9, 2038)
        swap = Swap(startDate,endDate,0.04,discCurve,projCurve)
        #
        exercise_date = ql.TARGET().advance(startDate, ql.Period('-2d'))
        sigma_n = 0.0060
        swaption = Swaption(swap, exercise_date, sigma_n)
        #
        print('')
        print('NPV:           %11.2f' % (swaption.npv()))
        print('NPV (manual):  %11.2f' % (swaption.npv_via_bachelier()))
        print('FairRate:      %11.6f' % (swaption.fairRate()))
        print('Annuity:       %11.2f' % (swaption.annuity()))
        print('Vega:          %11.2f' % (swaption.vega()))
        #
        pprint(swaption.bond_option_details())
        #
        self.assertAlmostEqual(swaption.npv(), swaption.npv_via_bachelier(), places=16)
        

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestSwaption))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())

