#!/usr/bin/env python3
#
# Tests the single-exponential fit methods.
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

    def test_estimate_initial(self):

        def plot(x, y, p=None):  # pragma: no cover
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(x, y, label='data')
            if p is not None:
                plt.plot(x, p[0] + p[1] * np.exp(p[2] * x))
            plt.legend()
            plt.show()

        # Noise free
        a, b, c = 8, 2, 0.3
        x = np.linspace(1.5, 2.5, 2000)
        y = a + b * np.exp(c * x)
        p, q, r = expfit.estimate_initial_single(x, y)
        self.assertAlmostEqual(p, a, 7)
        self.assertAlmostEqual(q, b, 6)
        self.assertAlmostEqual(r, c, 7)

        a, b, c = -1e3, 5, -0.3
        x = np.linspace(0.3, 4, 200)
        y = a + b * np.exp(c * x)
        p, q, r = expfit.estimate_initial_single(x, y)
        self.assertAlmostEqual(p, a, 7)
        self.assertAlmostEqual(q, b, 3)
        self.assertAlmostEqual(r, c, 5)

        a, b, c = 2e2, 21, -0.7
        x = np.linspace(0, 0.5, 9)
        y = a + b * np.exp(c * x)
        p, q, r = expfit.estimate_initial_single(x, y)
        self.assertAlmostEqual(p, a, 10)
        self.assertAlmostEqual(q, b, 1)
        self.assertAlmostEqual(r, c, 2)

        # With noise
        rng = np.random.default_rng(7)
        a, b, c = 73, 1, 0.18
        n = 1003
        x = np.linspace(0, 6.7, n)
        y = a + b * np.exp(c * x) + rng.normal(0, 0.05, n)
        p, q, r = expfit.estimate_initial_single(x, y)
        #plot(x, y, (p, q, r))
        self.assertAlmostEqual(p, a, 0)
        self.assertAlmostEqual(q, b, 0)
        self.assertAlmostEqual(r, c, 1)

        a, b, c = -51, -7.2, 1000
        n = 900
        x = np.linspace(1e-3, 7e-3, n)
        y = a + b * np.exp(c * x) + rng.normal(0, 100, n)
        p, q, r = expfit.estimate_initial_single(x, y)
        #plot(x, y, (p, q, r))
        self.assertAlmostEqual(p, a, -2)
        self.assertAlmostEqual(q, b, 0)
        self.assertAlmostEqual(r, c, -1)

        a, b, c = 1, 1e13, -3
        n = 88
        x = np.linspace(10, 11, n)
        y = a + b * np.exp(c * x) + rng.normal(0, 0.02, n)
        #plot(x, y, (p, q, r))
        p, q, r = expfit.estimate_initial_single(x, y)
        self.assertAlmostEqual(p, a, 1)
        self.assertAlmostEqual(r, c, 0)
        self.assertLess(np.sum((y - p - q * np.exp(r * x))**2), 0.1)

        # Edge case: perfectly flat line, no noise
        x = np.linspace(0, 1, 10)
        y = 3 * np.ones(x.shape)
        p, q, r = expfit.estimate_initial_single(x, y)
        self.assertEqual((p, q, r), (3, 0, 0))

        # Flat line with noise: hit b < 1e-100
        x = np.linspace(0, 1, 3000)
        y = 3 * np.ones(x.shape) + rng.normal(0, 1e-6, x.shape)
        p, q, r = expfit.estimate_initial_single(x, y)
        self.assertAlmostEqual(p, 3, 7)
        self.assertEqual(q, 0)
        self.assertEqual(r, 0)

        # Straight line through origin, no noise
        x = np.linspace(0, 1, 10)
        y = 3 * x
        p, q, r = expfit.estimate_initial_single(x, y)
        #plot(x, y, (p, q, r))
        self.assertAlmostEqual(np.sum((y - p - q * np.exp(r * x))**2), 0)
        self.assertAlmostEqual(p, -q)

        # Straight line through origin, with noise
        x = np.linspace(0, 1, 99)
        y = 3 * x + rng.normal(0, 0.1, x.shape)
        p, q, r = expfit.estimate_initial_single(x, y)
        #plot(x, y, (p, q, r))
        self.assertAlmostEqual(p, -q, 1)
        self.assertLess(np.sum((y - p - q * np.exp(r * x))**2), 1)

        # Straight line with offset and noise
        x = np.linspace(0, 1, 99)
        y = 4 + 2 * x + rng.normal(0, 0.1, x.shape)
        p, q, r = expfit.estimate_initial_single(x, y)
        #plot(x, y, (p, q, r))
        self.assertAlmostEqual(p + q, 4, 0)
        self.assertLess(np.sum((y - p - q * np.exp(r * x))**2), 1.5)

        # Vets, but can be disabled
        a, b, c = 3, 5, -0.7
        x = np.linspace(0.5, 1.5, 100)
        y = a + b * np.exp(c * x)
        self.assertRaisesRegex(
            ValueError, 'must have same length, got 100 and 99',
            expfit.estimate_initial_single, x, y[:-1], 5)
        expfit.estimate_initial_single(x, y[:-1], 5, vet=False)

        self.assertRaisesRegex(
            ValueError, 'At least 3', expfit.estimate_initial_single,
            [1, 2], [3, 4], 1)

    def test_rmse_single(self):

        a, b, c = 1, 2, -3
        x = np.linspace(0, 1, 99)
        y = a + b * np.exp(c * x)
        self.assertAlmostEqual(expfit.rmse_single(x, y, a, b, c), 0, 14)
        y = 3 * np.ones(x.shape)
        self.assertEqual(
            expfit.rmse_single(x, y, 0, 0, c), np.sqrt(np.sum(y**2)))
        y = 10 + b * np.exp(c * x)
        self.assertAlmostEqual(
            expfit.rmse_single(x, y, a, b, c), np.sqrt(99 * 81), 14)

    def test_single_edge_cases(self):

        x = np.linspace(0, 1, 10)
        y = np.zeros(x.shape)   # Means scaling to unit square would div by 0
        a, b, c = expfit.fit_single(x, y)
        self.assertEqual(a, 0)
        self.assertEqual(b, 0)
        self.assertEqual(c, 0)

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
        if maxr is None and maxrmse is None:  # pragma: no cover
            raise ValueError('Either maxr or maxrmse must be set')

        t = np.linspace(t0, t0 + duration, n)
        vt = a + b * np.exp(c * t)
        s = max(fnoise * abs(vt[0] - vt[-1]), 1e-9)
        v = vt + self.r.normal(0, s, size=n)

        af, bf, cf = expfit.fit_single(t, v, plot=plot)
        rt = expfit.rmse_single(t, v, a, b, c)
        rf = expfit.rmse_single(t, v, af, bf, cf)

        if plot:  # pragma: no cover
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
        sos(-10, -2, 9, 2, 600, maxr=1.01, fnoise=1, plot=plot)

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

        if plot:  # pragma: no cover
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

        if plot:  # pragma: no cover
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

    def test_single_tau(self):

        a, b, c = 3, -1, 3
        t = np.linspace(0, 10, 10)
        v = a + b * np.exp(-t / c)
        r = expfit.fit_single_tau(t, v)
        self.assertAlmostEqual(r, 3, 6)

        # Negative infinity
        a, b, c = 1, 0, 3
        t = np.linspace(0, 10, 10)
        v = a + b * np.exp(-t / c)
        r = expfit.fit_single_tau(t, v)
        self.assertTrue(np.isinf(r))
        self.assertLess(r, 0)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
