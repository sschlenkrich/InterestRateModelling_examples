import sys
sys.path.append('./')

import unittest

from test_bermudan_option import TestBermudanOption
from tests.test_helpers import TestHelpers
from tests.test_hull_white_model import TestHullWhiteModel
from tests.test_methods import TestPricingMethods
from tests.test_monte_carlo_simulation import TestMonteCarloSimulation
from tests.test_sabr_model import TestSabrModel
from tests.test_swap import TestSwap
from tests.test_swaption import TestSwaption
from tests.test_yieldcurve import TestYieldCurve


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestBermudanOption))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestHelpers))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestHullWhiteModel))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestPricingMethods))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestMonteCarloSimulation))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestSabrModel))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestSwap))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestSwaption))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestYieldCurve))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
