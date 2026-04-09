#!/usr/bin/env python3
#
# Tests the WHAT DOES IT DOOOOOOOOOOOOOOOO
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import unittest

import numpy as np

import expfit


class TestSingle(unittest.TestCase):
    """
    Tests fitting of single exponentials.
    """
    @classmethod
    def setUpClass(cls):
        # Create in each test and seed!
        cls.r = None

    def single_on_single(self, a, b, c, duration, n, fnoise=0.01, t0=0,
                         maxr=1, maxrmse=None, plot=False):
        """
        Fits an exponential to a signal containing an exponential, and returns
        the ratio ``RMSE(fit, noisy) / RMSE(true, noisy)``.

        Creates a signal ``a + b exp(c t)`` with ``n`` points from ``t0`` to
        ``t0 + duration``, and normally distributed noise with a variance of
        ``fnoise`` times the clea signal's magnitude.

        Then fits a signal, and calculates the RMSEs between (1) the noisy
        signal and the exponential with the input parameters, and (2) the
        noisy signal and the fit. For a perfect fit the ratio Rfit/Rtrue will
        be less than 1, as the fit will incorporate some of the bias introduced
        by the random noise.
        """
        if maxr is None and maxrmse is None:
            raise ValueError('Either maxr or maxrmse must be set')

        t = np.linspace(t0, t0 + duration, n)
        vt = a + b * np.exp(c * t)
        s = max(fnoise * abs(vt[0] - vt[-1]), 1e-9)
        v = vt + self.r.normal(0, s, size=n)

        af, bf, cf = expfit.fit_single(t, v, plot=plot)
        rt = expfit.rmse_single(t, v, a, b, c)
        rf = expfit.rmse_single(t, v, af, bf, cf)

        if plot:
            print(f'True: {a:+.5e} {b:+.5e} {c:+.5e}')
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

    def single_on_double(self, a, b, c, d, e, duration=1, n=100, fnoise=0.01,
                         t0=0, maxr=2, maxrmse=1, plot=False):
        """
        Fits a single exponential to a signal containing a double exponential.
        """
        t = np.linspace(t0, t0 + duration, n)
        vt = a + b * np.exp(c * t) + d * np.exp(e * t)
        s = max(fnoise * abs(vt[0] - vt[-1]), 1e-9)
        v = vt + self.r.normal(0, s, size=n)

        af, bf, cf = expfit.fit_single(t, v, plot=plot)
        rf = expfit.rmse_single(t, v, af, bf, cf)

        # Dominant rate
        bdom, cdom = [(b, c), (d, e)][np.argmax(np.abs((c, e)))]
        rdom = cf / cdom

        if plot:
            print(f'True:              {bdom:+.5e} {cdom:+.5e}')
            print(f'Full: a {a:+.5e}')
            print(f'      b {b:+.5e} c {c:+.5e}')
            print(f'      d {d:+.5e} e {e:+.5e}')
            print(f'RMSE fit: {rf}')
            print(f'Estimate / dominant true: {rdom:.5e} ({1 / rdom:.5e})')

            import matplotlib.pyplot as plt
            plt.show()

        with self.subTest(a=a, b=b, c=c, d=d, e=e, duration=duration, n=n,
                          fnoise=fnoise, t0=t0):
            self.assertLess(rdom, maxr)
            self.assertGreater(rdom, 1 / maxr)
            self.assertLess(rf, maxrmse)

    def single_on_triple(self, a, b, c, d, e, f, g, duration=1, n=100,
                         fnoise=0.01, t0=0, maxr=2, maxrmse=2, plot=False):
        """
        Fits a single exponential to a signal containing a double exponential.
        """
        t = np.linspace(t0, t0 + duration, n)
        vt = a + b * np.exp(c * t) + d * np.exp(e * t) + f * np.exp(g * t)
        s = max(fnoise * abs(vt[0] - vt[-1]), 1e-9)
        v = vt + self.r.normal(0, s, size=n)

        af, bf, cf = expfit.fit_single(t, v, plot=plot)
        rf = expfit.rmse_single(t, v, af, bf, cf)

        # Dominant rate
        bdom, cdom = [(b, c), (d, e), (f, g)][np.argmax(np.abs((c, e, g)))]
        rdom = cf / cdom

        if plot:
            print(f'True:              {bdom:+.5e} {cdom:+.5e}')
            print(f'Full: a {a:+.5e}')
            print(f'      b {b:+.5e} c {c:+.5e}')
            print(f'      d {d:+.5e} e {e:+.5e}')
            print(f'      f {f:+.5e} g {g:+.5e}')
            print(f'RMSE fit: {rf}')
            print(f'Estimate / dominant true: {rdom:.5e} ({1 / rdom:.5e})')

            import matplotlib.pyplot as plt
            plt.show()

        with self.subTest(a=a, b=b, c=c, d=d, e=e, duration=duration, n=n,
                          fnoise=fnoise, t0=t0):
            self.assertLess(rdom, maxr)
            self.assertGreater(rdom, 1 / maxr)
            self.assertLess(rf, maxrmse)

    def test_single_on_single(self):
        # Test single exponentials on single exponential data
        sos = self.single_on_single
        self.r = np.random.default_rng(1)
        plot = False

        # Moderate
        sos(0, -1, 3, 2, 123, plot=plot)
        sos(3e2, 2, 4, 2, 200, plot=plot)
        sos(5e3, 3, -0.5, 5, 500, plot=plot)
        sos(-1e3, 10, -9, 2, 50, plot=plot)

        # Steep
        sos(4e5, -1, 30, 2, 300, maxr=1.2, plot=plot)
        sos(-1e3, 10, -9, 2, 1000, plot=plot)
        sos(3e5, -1, 15, 2, 500, maxr=1.02, plot=plot)

        # Almost straight
        sos(3, -1, 0.3, 2, 3000, plot=plot)
        sos(-6e2, +1, 0.03, 2, 3000, plot=plot)
        sos(0, 1, 1e-6, 1, 200, maxr=1.001, plot=plot)
        sos(1, 2, 1e-6, 1, 200, fnoise=0.2, plot=plot)

        # Flat
        sos(1, 0, 3, 1, 200, maxr=1.1, plot=plot)

        # Clean
        sos(0, -1, 3, 2, 123, fnoise=0, maxr=None, maxrmse=0.05, plot=plot)
        sos(4e2, 2, 4, 2, 1000, fnoise=1e-3, maxr=1.01, plot=plot)
        sos(7e3, 3, -0.5, 5, 500, fnoise=1e-2, plot=plot)

        # Noisy
        sos(4, 10, 3, 2, 100, fnoise=0.11, plot=plot)
        sos(4, 10, 3, 2, 100, fnoise=0.3, plot=plot)
        sos(51, -1, -0.5, 5, 200, maxr=1.01, fnoise=0.5, plot=plot)
        sos(-10, -2, 9, 2, 600, maxr=1.01, fnoise=1,  plot=plot)

        # Short
        sos(30, 2, 4, 2, 10, plot=plot)
        sos(15, 3, 5, 1.5, 9, plot=plot)
        sos(10, -3, -1e3, 5, 8, plot=plot)
        sos(-2, 3, 0.05, 5, 6, plot=plot)
        sos(-30, 10, -5, 0.2, 5, plot=plot)
        sos(20, -10, 7, 2, 4, maxr=1.4, plot=plot)
        sos(-5, 10, -2, 4, 3, plot=plot)

        # Dense
        sos(1e2, -2, 3, 2, 10000, plot=plot)
        sos(1e3, 8, -0.12, 5, 100000, plot=plot)

    def test_single_on_double(self):
        # Test single exponentials on single exponential data
        sod = self.single_on_double
        self.r = np.random.default_rng(2)
        plot = False

        # Same direction
        sod(0, -1, 3, -4, 5, maxr=1.2, maxrmse=60, plot=plot)
        sod(0, -1, 3, -2, 5, maxr=1.1, maxrmse=35, plot=plot)
        sod(0, -1, 3, -1, 5, maxr=1.1, maxrmse=60, plot=plot)
        sod(0, -1, 3, -0.5, 5, maxr=1.2, maxrmse=10, plot=plot)
        sod(0, -1, 3, -1e-6, 5, maxr=1.7, maxrmse=3, plot=plot)
        sod(0, -1, 3, -1e-12, 5, maxr=1.7, maxrmse=2, plot=plot)
        sod(0, 1, -3, 1, -3.1, maxr=1.1, maxrmse=1, plot=plot)
        sod(0, 2, -3, 1, -2.8, maxr=1.1, maxrmse=1, plot=plot)
        sod(0, 2, -3, 1, -0.02, maxr=1.1, maxrmse=1, plot=plot)

    def test_single_on_triple(self):
        # Test single exponentials on single exponential data
        sot = self.single_on_triple
        self.r = np.random.default_rng(2)
        plot = False

        # Same direction
        sot(0, -6, -0.1, -3, -10, -2, -2, maxr=2.5, maxrmse=3, plot=plot)
        sot(0, -6, 0.1, -3, 10, -2, 2, maxr=1.01, maxrmse=7e3, plot=plot)
        sot(0, 3, -1, 3, -6, 2, -2, maxr=2, maxrmse=1, plot=plot)
        sot(0, 4, 0.2, 2.8, 10, 1.1, 20, maxr=1.1, maxrmse=7e8, plot=plot)

        # Opposing direction, over fast
        #sot(0, 6, -160, -3, -10, 0, 0, n=50, maxrmse=3, plot=True)
        #sot(0, 6, -160, -3, -10, 0, 0, n=1000, maxrmse=3, plot=True)
        # Needs user to chop it off


if __name__ == '__main__':
    unittest.main()
