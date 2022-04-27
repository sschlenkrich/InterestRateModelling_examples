import sys
sys.path.insert(0, "./")

import unittest

from tests.test_helpers import TestHelpers
from tests.test_swap import TestSwap
from tests.test_yieldcurve import TestYieldCurve


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestHelpers))
    suite.addTest(unittest.makeSuite(TestSwap))
    suite.addTest(unittest.makeSuite(TestYieldCurve))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
