
import sys
sys.path.append('./')

import numpy as np
import matplotlib.pyplot as plt
import unittest

from src.sabr_model import SabrModel

class TestSabrModel(unittest.TestCase):
    """
    Test SABR model construction and Vanilla option pricing.
    """

    def test_sabr_model_setup(self):
        model = SabrModel(0.05,1.0,0.0420,0.5000,0.5000,0.7)
        model.calibrate_atm(0.01)
        strikes = np.linspace(0.001, 0.100, 200)
        vols = np.array([ model.normal_volatility(K) for K in strikes ])
        densities = np.array([ model.density(K) for K in strikes ])
        fig, ax1 = plt.subplots()
        ax1.plot(strikes, vols)
        ax2=ax1.twinx()
        ax2.plot(strikes, densities)
        ax1.set_xlabel('strike / swap raate')
        ax1.set_ylabel('Normal implied volatility')
        ax2.set_ylabel('density')
        # plt.show()


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestSabrModel))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())

