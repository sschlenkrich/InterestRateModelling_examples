import sys
sys.path.append('./')

import unittest

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
    suite.addTest(unittest.makeSuite(TestHelpers))
    suite.addTest(unittest.makeSuite(TestHullWhiteModel))
    suite.addTest(unittest.makeSuite(TestPricingMethods))
    suite.addTest(unittest.makeSuite(TestMonteCarloSimulation))
    suite.addTest(unittest.makeSuite(TestSabrModel))
    suite.addTest(unittest.makeSuite(TestSwap))
    suite.addTest(unittest.makeSuite(TestSwaption))
    suite.addTest(unittest.makeSuite(TestYieldCurve))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
