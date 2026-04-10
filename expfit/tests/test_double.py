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

    def double_decaying_on_double(self, a, b, c, d, e, duration=2, n=200,
                                  fnoise=0.01, t0=0, digits=[], plot=False):
        # maxr=1, maxrmse=None,
        """
        Tests a double exponential fit to a double exponential signal.

        Criteria: ``digits`` is the ``assertAlmostEqual`` precision with which
        fitted parameters match.
        """
        t = np.linspace(t0, t0 + duration, n)
        vt = a + b * np.exp(c * t) + d * np.exp(e * t)
        s = max(fnoise * abs(vt[0] - vt[-1]), 1e-9)
        v = vt + self.r.normal(0, s, size=n)

        plot_params = (a, b, c, d, e) if plot else False
        af, bf, cf, df, ef = expfit.fit_double_decaying(t, v, plot=plot_params)
        rt = expfit.rmse_double(t, v, a, b, c, d, e)
        rf = expfit.rmse_double(t, v, af, bf, cf, df, ef)

        if plot:  # pragma: no cover
            print(f'True: {a:+.5e} {b:+.5e} {c:+.5e} {d:+.5e} {e:+.5e}')
            print(f'RMSE true: {rt}')
            print(f'RMSE fit:  {rf}')
            print(f'ratio: {rf / rt}')
            import matplotlib.pyplot as plt
            plt.show()

        with self.subTest(a=a, b=b, c=c, d=d, e=e, duration=duration, n=n,
                          fnoise=fnoise, t0=t0):
            if len(digits) == 5:
                self.assertAlmostEqual(af, a, digits[0])
                self.assertAlmostEqual(bf, b, digits[1])
                self.assertAlmostEqual(cf, c, digits[2])
                self.assertAlmostEqual(df, d, digits[3])
                self.assertAlmostEqual(ef, e, digits[4])
            else:  # pragma: no cover
                raise ValueError('No test criteria set')

    def test_double_on_double(self):
        # Test double exponentials on double exponential data
        dod = self.double_decaying_on_double
        self.r = np.random.default_rng(5)
        plot = False

        # Both decaying
        dod(200, 4, -5, 100, -3, digits=(0, -1, 0, -1, -1), plot=plot)
        dod(200, 4, -5, 10, -2, duration=1, digits=(0, -1, -1, -1, 2),
            plot=plot)
        dod(200, 4, -5, 1, -0.5, digits=(0, 0, 0, 1, 0), plot=plot)
        dod(200, 4, -5, 10, -1, duration=1, digits=(-2, -1, -1, -3, -1),
            plot=plot)

        # Both expanding
        #dod(200, 4, -5, 1, -3, plot=True)

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

    def test_double_edge_cases(self):

        x = np.linspace(0, 1, 10)
        y = np.zeros(x.shape)   # Means scaling to unit square would div by 0
        a, b, c, d, e = expfit.fit_double_decaying(x, y)
        self.assertEqual(a, 0)
        self.assertEqual(b, 0)
        self.assertEqual(c, 0)
        self.assertEqual(d, 0)
        self.assertEqual(e, 0)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
