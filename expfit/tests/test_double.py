#!/usr/bin/env python3
#
# Tests the double-exponential fitting methods.
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import unittest

import numpy as np


import expfit

class TestDouble(unittest.TestCase):
    """
    Tests fitting of double exponentials.
    """
    @classmethod
    def setUpClass(cls):
        # Create in each test and seed!
        cls.r = None

    def double_on_double(self, a, b, c, d, e, duration=1, n=200, fnoise=0.01,
                         t0=0, plot=False):
        # maxr=1, maxrmse=None,
        """

        """
        #if maxr is None and maxrmse is None:
        #    raise ValueError('Either maxr or maxrmse must be set')

        t = np.linspace(t0, t0 + duration, n)
        vt = a + b * np.exp(c * t) + d * np.exp(e * t)
        s = max(fnoise * abs(vt[0] - vt[-1]), 1e-9)
        v = vt + self.r.normal(0, s, size=n)

        af, bf, cf, df, ef = expfit.fit_double(t, v, plot=plot)
        rt = expfit.rmse_double(t, v, a, b, c, d, e)
        rf = expfit.rmse_double(t, v, af, bf, cf, df, ef)

        if plot:
            print(f'True: {a:+.5e} {b:+.5e} {c:+.5e} {d:+.5e} {e:+.5e}')
            print(f'RMSE true: {rt}')
            print(f'RMSE fit:  {rf}')
            print(f'ratio: {rf / rt}')
            import matplotlib.pyplot as plt
            plt.show()

        with self.subTest(a=a, b=b, c=c, duration=duration, n=n, fnoise=fnoise,
                          t0=t0):
            if maxr is not None:
                self.assertLess(rf / rt, maxr)
            if maxrmse is not None:
                self.assertLess(rf, maxrmse)

    def test_single_on_double(self):
        # Test single exponentials on single exponential data
        dod = self.double_on_double
        self.r = np.random.default_rng(5)
        plot = True

        dod(0, -1, 3, -4, 5, plot=True)

        # Same direction
        #dod(0, -1, 3, -4, 5, maxr=1.2, maxrmse=60, plot=plot)
        #dod(0, -1, 3, -2, 5, maxr=1.1, maxrmse=35, plot=plot)
        #dod(0, -1, 3, -1, 5, maxr=1.1, maxrmse=60, plot=plot)
        #dod(0, -1, 3, -0.5, 5, maxr=1.2, maxrmse=10, plot=plot)
        #dod(0, -1, 3, -1e-6, 5, maxr=1.7, maxrmse=3, plot=plot)
        #dod(0, -1, 3, -1e-12, 5, maxr=1.7, maxrmse=2, plot=plot)
        #dod(0, 1, -3, 1, -3.1, maxr=1.1, maxrmse=1, plot=plot)
        #dod(0, 2, -3, 1, -2.8, maxr=1.1, maxrmse=1, plot=plot)
        #dod(0, 2, -3, 1, -0.02, maxr=1.1, maxrmse=1, plot=plot)


if __name__ == '__main__':
    unittest.main()
