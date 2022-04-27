from calendar import c
import sys
sys.path.insert(0, "./")

import numpy as np
import unittest

from src.helpers import black
from src.helpers import black_implied_vol
from src.helpers import bachelier
from src.helpers import bachelier_implied_vol


class TestHelpers(unittest.TestCase):
    """
    Test Black and Bachelier formula.
    """

    def test_black(self):
        #
        F = 0.03
        K = 0.02
        sigma = 0.25
        T = 2.0
        callOrPut = -1 # put
        fwd_price = black(K, F, sigma, T, callOrPut)
        impl_vol = black_implied_vol(fwd_price, K, F, T, callOrPut)
        self.assertAlmostEqual(impl_vol, sigma, places=8)

    def test_bachelier(self):
        #
        F = 0.03
        K = 0.02
        sigma = 0.01
        T = 2.0
        callOrPut = -1 # put
        fwd_price = bachelier(K, F, sigma, T, callOrPut)
        impl_vol = bachelier_implied_vol(fwd_price, K, F, T, callOrPut)
        self.assertAlmostEqual(impl_vol, sigma, places=8)



if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestHelpers))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())

