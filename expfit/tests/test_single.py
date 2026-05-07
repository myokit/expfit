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

        rng = np.random.default_rng(71)
        plot = False

        def estimate(x, y, transform=True, plot=False):
            ret = expfit.estimate_initial_single(
                x, y, transform=transform, plot=plot)
            if plot:
                import matplotlib.pyplot as plt
                plt.show()
            return ret

        # Noise free
        a, b, c = 8, 2, 0.3
        x = np.linspace(1.5, 2.5, 2000)
        y = a + b * np.exp(c * x)
        p, q, r = estimate(x, y, plot=plot)
        self.assertAlmostEqual(p, a, delta=1e-9)
        self.assertAlmostEqual(q, b, delta=1e-7)
        self.assertAlmostEqual(r, c, delta=1e-8)

        a, b, c = -1000, 5, -0.3
        x = np.linspace(0.3, 4, 200)
        y = a + b * np.exp(c * x)
        p, q, r = estimate(x, y, plot=plot)
        self.assertAlmostEqual(p, a, delta=1e-10)
        self.assertAlmostEqual(q, b, delta=2e-4)
        self.assertAlmostEqual(r, c, delta=1e-5)

        a, b, c = 200, 21, -0.7
        x = np.linspace(0, 0.5, 9)
        y = a + b * np.exp(c * x)
        p, q, r = estimate(x, y, plot=plot)
        self.assertAlmostEqual(p, a, delta=1e-11)
        self.assertAlmostEqual(q, b, delta=0.1)
        self.assertAlmostEqual(r, c, delta=1e-3)

        # With noise
        a, b, c = 73, 1, 0.18
        n = 1003
        x = np.linspace(0, 6.7, n)
        y = a + b * np.exp(c * x) + rng.normal(0, 0.05, n)
        p, q, r = estimate(x, y, plot=plot)
        self.assertAlmostEqual(p, a, delta=0.2)
        self.assertAlmostEqual(q, b, delta=0.2)
        self.assertAlmostEqual(r, c, delta=0.02)

        a, b, c = -51, -7.2, 1000
        n = 900
        x = np.linspace(1e-3, 7e-3, n)
        y = a + b * np.exp(c * x) + rng.normal(0, 100, n)
        p, q, r = estimate(x, y, plot=plot)
        self.assertAlmostEqual(p, a, delta=6)
        self.assertAlmostEqual(q, b, delta=2)
        self.assertAlmostEqual(r, c, delta=40)

        a, b, c = 1, 1e13, -3
        n = 88
        x = np.linspace(10, 11, n)
        y = a + b * np.exp(c * x) + rng.normal(0, 0.02, n)
        p, q, r = estimate(x, y, plot=plot)
        self.assertAlmostEqual(p, a, delta=0.02)
        self.assertAlmostEqual(q, b, delta=8e12)
        self.assertAlmostEqual(r, c, delta=0.1)
        self.assertLess(expfit.rmse_single(x, y, p, q, r), 0.1)

        # Edge case: perfectly flat line, no noise
        x = np.linspace(0, 1, 10)
        y = 3 * np.ones(x.shape)
        p, q, r = estimate(x, y, plot=plot)
        self.assertEqual((p, q, r), (3, 0, 0))

        # Flat line with noise
        x = np.linspace(0, 1, 3000)
        y = 3 * np.ones(x.shape) + rng.normal(0, 1e-9, x.shape)
        p, q, r = estimate(x, y, plot=plot)
        self.assertAlmostEqual(p, 3, delta=1e-10)
        self.assertAlmostEqual(q, 0, delta=1e-11)

        # Straight line through origin, no noise
        # Note: the transform amplifies the numerical noise here, causing the
        # RMSE to be non-zero for the transformed case.
        x = np.linspace(0, 1, 10)
        y = 3 * x
        p, q, r = estimate(x, y, plot=plot)
        self.assertLess(expfit.rmse_single(x, y, p, q, r), 0.6)
        self.assertEqual(p, -q)
        p, q, r = estimate(x, y, transform=False, plot=plot)
        self.assertEqual(expfit.rmse_single(x, y, p, q, r), 0)
        self.assertEqual(p, -q)

        # Straight line through origin, with noise
        x = np.linspace(0, 1, 99)
        y = 3 * x + rng.normal(0, 0.1, x.shape)
        p, q, r = estimate(x, y, plot=plot)
        self.assertAlmostEqual(p, -q, delta=0.1)
        self.assertLess(expfit.rmse_single(x, y, p, q, r), 0.1)

        # Straight line with offset and noise
        x = np.linspace(0, 1, 99)
        y = 4 + 2 * x + rng.normal(0, 0.1, x.shape)
        p, q, r = estimate(x, y, plot=plot)
        self.assertAlmostEqual(p + q, 4, delta=0.5)
        self.assertLess(expfit.rmse_single(x, y, p, q, r), 0.2)

        # Vets, but can be disabled
        a, b, c = 3, 5, -0.7
        x = np.linspace(0.5, 1.5, 100)
        y = a + b * np.exp(c * x)
        self.assertRaisesRegex(
            ValueError, 'must have same length, got 100 and 99',
            expfit.estimate_initial_single, x, y[:-1])
        self.assertRaisesRegex(
            ValueError, 'could not be broadcast together with shapes',
            expfit.estimate_initial_single, x, y[:-1], vet=False)

        self.assertRaisesRegex(
            ValueError, 'At least 3', expfit.estimate_initial_single,
            [1, 2], [3, 4], 1)

    def test_estimate_initial_straight(self):
        # Edge cases: straight and flat lines for estimate_initial_single

        rng = np.random.default_rng(71)
        plot = False

        def estimate(x, y, transform=True, plot=False):
            ret = expfit.estimate_initial_single(
                x, y, transform=transform, plot=plot)
            if plot:
                import matplotlib.pyplot as plt
                plt.show()
            return ret

        # Edge case: perfectly flat line, no noise
        x = np.linspace(0, 1, 10)
        y = 3 * np.ones(x.shape)
        p, q, r = estimate(x, y, plot=plot)
        self.assertEqual((p, q, r), (3, 0, 0))

        # Flat line with noise
        x = np.linspace(0, 1, 3000)
        y = 3 * np.ones(x.shape) + rng.normal(0, 1e-9, x.shape)
        p, q, r = estimate(x, y, plot=plot)
        self.assertAlmostEqual(p, 3, delta=1e-10)
        self.assertAlmostEqual(q, 0, delta=1e-11)

        # Straight line through origin, no noise
        # Note: the transform amplifies the numerical noise here, causing the
        # RMSE to be non-zero for the transformed case.
        x = np.linspace(0, 1, 10)
        y = 3 * x
        p, q, r = estimate(x, y, plot=plot)
        self.assertLess(expfit.rmse_single(x, y, p, q, r), 0.6)
        self.assertEqual(p, -q)
        p, q, r = estimate(x, y, transform=False, plot=True)
        self.assertEqual(p, np.mean(y))
        self.assertEqual(q, 0)
        self.assertEqual(r, 0)

        # Straight line through origin, with noise
        x = np.linspace(0, 1, 99)
        y = 3 * x + rng.normal(0, 0.1, x.shape)
        p, q, r = estimate(x, y, plot=plot)
        self.assertAlmostEqual(p, -q, delta=0.1)
        self.assertLess(expfit.rmse_single(x, y, p, q, r), 0.1)

        # Straight line with offset and noise
        x = np.linspace(0, 1, 99)
        y = 4 + 2 * x + rng.normal(0, 0.1, x.shape)
        p, q, r = estimate(x, y, plot=plot)
        self.assertAlmostEqual(p + q, 4, delta=0.5)
        self.assertLess(expfit.rmse_single(x, y, p, q, r), 0.2)

        rng = np.random.default_rng(2)
        n = 200
        x = np.linspace(0, 1, n)
        y = 3 * np.zeros(x.shape)
        y += rng.normal(0, 1, x.shape)
        p, q, r = estimate(x, y, plot=plot)
        self.assertAlmostEqual(p, np.mean(y))
        self.assertEqual(q, 0)
        self.assertEqual(r, 0)

    def test_single_error(self):

        # TODO
        '''
        a, b, c = 1, 2, -3
        x = np.linspace(0, 1, 99)
        y = a + b * np.exp(c * x)
        self.assertAlmostEqual(expfit.rmse_single(x, y, a, b, c), 0, 14)
        y = 3 * np.ones(x.shape)
        self.assertEqual(
            expfit.rmse_single(x, y, 0, 0, c), np.sqrt(np.sum(y**2) / len(x)))
        y = 10 + b * np.exp(c * x)
        self.assertAlmostEqual(
            expfit.rmse_single(x, y, a, b, c), np.sqrt(81), 14)
        '''

    def test_single_edge_cases(self):
        # Test for a specific divide by zero case

        x = np.linspace(0, 1, 10)
        y = np.zeros(x.shape)   # Means scaling to unit square would div by 0
        a, b, c = expfit.fit_single(x, y)
        self.assertEqual(a, 0)
        self.assertEqual(b, 0)
        self.assertEqual(c, 0)

    def single_on_single(self, a, b, c, duration, n, fnoise=0.01, t0=0,
                         deltas=[], ratio=1, rmse=None, plot=False):
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

        Criteria: ``deltas`` are the ``assertAlmostEqual`` ``delta`` of the
        parameters, ``ratio`` is the max rmse fit/true ratio, and ``rmse`` is
        the max rmse.
        """
        t = np.linspace(t0, t0 + duration, n)
        vt = a + b * np.exp(c * t)
        s = max(fnoise * abs(vt[0] - vt[-1]), 1e-9)
        v = vt + self.r.normal(0, s, size=n)

        af, bf, cf = expfit.fit_single(t, v, plot=(a, b, c) if plot else False)
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
            if len(deltas) == 3:
                if plot:
                    print(abs(af - a))
                    print(abs(bf - b))
                    print(abs(cf - c))
                self.assertAlmostEqual(af, a, delta=deltas[0])
                self.assertAlmostEqual(bf, b, delta=deltas[1])
                self.assertAlmostEqual(cf, c, delta=deltas[2])
            if ratio is not None:
                self.assertLess(rf / rt, ratio)
            if rmse is not None:
                self.assertLess(rf, rmse)
            if len(deltas) != 3 and ratio is None and rmse is None:
                raise Exception('No test criteria set')  # pragma: no cover

    def test_single_on_single_basic(self):
        # Test single exponentials on single exponential data
        sos = self.single_on_single
        self.r = np.random.default_rng(1)
        plot = False

        # Moderate
        sos(0, -1, 3, 2, 123, deltas=(0.5, 0.1, 0.1), plot=plot)
        sos(3e2, 2, 4, 2, 200, deltas=(10, 0.1, 0.01), plot=plot)
        sos(5e3, 3, -0.5, 5, 500, deltas=(0.05, 0.05, 0.01), plot=plot)
        sos(-1e3, 10, -9, 2, 50, deltas=(0.1, 0.1, 0.1), plot=plot)

        # Steep: These rely more on the guess than on the fitting
        # These would benefit from a de-steeper
        sos(4e5, -1, 30, 2, 300, ratio=1.03, plot=plot)
        sos(-1e3, 10, -9, 2, 1000, deltas=(1e-2, 0.1, 0.1), plot=plot)
        sos(3e5, -1, 15, 2, 500, plot=plot)

        # Almost straight
        sos(3, -1, 0.3, 2, 3000, deltas=(0.1, 0.1, 1e-2), plot=plot)
        sos(-6e2, +1, 0.03, 2, 3000, ratio=1.001, deltas=(0.5, 0.5, 0.1),
            plot=plot)
        sos(0, 1, 1e-6, 1, 200, ratio=1, plot=plot)
        sos(1, 2, 1e-6, 1, 200, fnoise=0.2, plot=plot)

        # Both sides of zero
        sos(-2.5, 5, -2, 2, 50, deltas=(0.1, 1e-2, 0.1), plot=plot)

    def test_single_on_single_straight(self):
        # Test single exponentials on single exponential data
        sos = self.single_on_single
        self.r = np.random.default_rng(1)
        plot = False

        # Flat: extremely dependent on random seed
        sos(1, 0, 3, 1, 200, plot=plot)
        sos(1, 0, 3, 1, 200, plot=plot)
        sos(1, 0, 3, 1, 200, plot=plot)

    def test_single_on_single_noisy(self):
        # Test single exponentials on single exponential data
        sos = self.single_on_single
        self.r = np.random.default_rng(12)
        plot = False

        # Clean
        sos(0, -1, 3, 2, 123, fnoise=0,
            deltas=(1e-9, 1e-11, 1e-11), ratio=None, rmse=4e-6, plot=plot)
        sos(4e2, 2, 4, 2, 1000, fnoise=1e-3,
            deltas=(0.1, 2e-3, 5e-4), plot=plot)
        sos(7e3, 3, -0.5, 5, 500, fnoise=1e-2,
            deltas=(0.01, 1e-3, 1e-2), plot=plot)

        # Noisy
        sos(4, 10, 3, 2, 100, fnoise=0.11, deltas=(100, 10, 0.1), plot=plot)
        sos(4, 10, 3, 2, 100, fnoise=0.3, deltas=(1000, 100, 1), plot=plot)
        sos(51, -1, -0.5, 5, 200, fnoise=0.5,
            deltas=(1, 0.1, 0.5), plot=plot)
        sos(-10, -2, 9, 2, 600, fnoise=1, deltas=(1e7, 1e5, 10), plot=plot)

    def test_single_on_single_dense(self):
        # Test single exponentials on single exponential data
        sos = self.single_on_single
        self.r = np.random.default_rng(1)
        plot = False

        # Short
        sos(30, 2, 4, 2, 10, deltas=(15, 0.1, 0.01), plot=plot)
        sos(15, 3, 5, 1.5, 9, deltas=(5, 0.5, 0.05), plot=plot)
        sos(10, -3, -1e3, 5, 8, deltas=(0.1, 0.01, 1e3), plot=plot)
        sos(-2, 3, 0.05, 5, 6, deltas=(5, 5, 0.1), plot=plot)
        sos(-30, 10, -5, 0.2, 5, deltas=(1, 1, 1), plot=plot)
        sos(20, -10, 7, 2, 4, deltas=(1e6, 1e3, 1e2), plot=plot)
        sos(-5, 10, -2, 4, 3, deltas=(1e-2, 1e-3, 0.5), plot=plot)

        # Dense
        sos(1e2, -2, 3, 2, 10000, deltas=(0.1, 5e-3, 5e-4), plot=plot)
        sos(1e3, 8, -0.12, 5, 100000, deltas=(1e-2, 1e-2, 1e-3), plot=plot)

    def single_on_double(self, a, b, c, d, e, duration=1, n=100, fnoise=0.01,
                         t0=0, rdom=2, rmse=1, plot=False):
        """
        Fits a single exponential to a signal containing a double exponential.

        Criteria: ``rdom`` is the ratio between estimated c and dominant c in
        given parameters, ``rmse`` is the max rmse.
        """
        t = np.linspace(t0, t0 + duration, n)
        vt = a + b * np.exp(c * t) + d * np.exp(e * t)
        s = max(fnoise * abs(vt[0] - vt[-1]), 1e-9)
        v = vt + self.r.normal(0, s, size=n)

        af, bf, cf = expfit.fit_single(t, v, plot=plot)
        rf = expfit.rmse_single(t, v, af, bf, cf)

        # Dominant rate
        bdom, cdom = [(b, c), (d, e)][np.argmax(np.abs((c, e)))]
        dr = cf / cdom

        if plot:  # pragma: no cover
            print(f'True:              {bdom:+.5e} {cdom:+.5e}')
            print(f'Full: a {a:+.5e}')
            print(f'      b {b:+.5e} c {c:+.5e}')
            print(f'      d {d:+.5e} e {e:+.5e}')
            print(f'RMSE fit: {rf}')
            print(f'Estimate / dominant true: {dr:.5e} ({1 / dr:.5e})')

            import matplotlib.pyplot as plt
            plt.show()

        with self.subTest(a=a, b=b, c=c, d=d, e=e, duration=duration, n=n,
                          fnoise=fnoise, t0=t0):
            self.assertLess(dr, rdom)
            self.assertGreater(dr, 1 / rdom)
            self.assertLess(rf, rmse)

    def test_single_on_double(self):
        # Test single exponentials on single exponential data
        sod = self.single_on_double
        self.r = np.random.default_rng(2)
        plot = False

        # Same direction
        sod(0, -1, 3, -4, 5, rdom=1.01, rmse=6, plot=plot)
        sod(0, -1, 3, -2, 5, rdom=1.1, rmse=3.1, plot=plot)
        sod(0, -1, 3, -1, 5, rdom=1.1, rmse=2, plot=plot)
        sod(0, -1, 3, -0.5, 5, rdom=1.2, rmse=1, plot=plot)
        sod(0, -1, 3, -1e-6, 5, rdom=1.7, rmse=0.2, plot=plot)
        sod(0, -1, 3, -1e-12, 5, rdom=1.7, rmse=0.2, plot=plot)
        sod(0, 1, -3, 1, -3.1, rdom=1.1, rmse=0.02, plot=plot)
        sod(0, 2, -3, 1, -2.8, rdom=1.1, rmse=0.04, plot=plot)
        sod(0, 2, -3, 1, -0.02, rdom=1.1, rmse=0.02, plot=plot)

    def single_on_triple(self, a, b, c, d, e, f, g, duration=1, n=100,
                         fnoise=0.01, t0=0, rdom=2, rmse=2, plot=False):
        """
        Fits a single exponential to a signal containing a double exponential.

        Criteria: ``rdom`` is the ratio between estimated c and dominant c in
        given parameters, ``rmse`` is the max rmse.
        """
        t = np.linspace(t0, t0 + duration, n)
        vt = a + b * np.exp(c * t) + d * np.exp(e * t) + f * np.exp(g * t)
        s = max(fnoise * abs(vt[0] - vt[-1]), 1e-9)
        v = vt + self.r.normal(0, s, size=n)

        af, bf, cf = expfit.fit_single(t, v, plot=plot)
        rf = expfit.rmse_single(t, v, af, bf, cf)

        # Dominant rate
        bdom, cdom = [(b, c), (d, e), (f, g)][np.argmax(np.abs((c, e, g)))]
        dr = cf / cdom

        if plot:  # pragma: no cover
            print(f'True:              {bdom:+.5e} {cdom:+.5e}')
            print(f'Full: a {a:+.5e}')
            print(f'      b {b:+.5e} c {c:+.5e}')
            print(f'      d {d:+.5e} e {e:+.5e}')
            print(f'      f {f:+.5e} g {g:+.5e}')
            print(f'RMSE fit: {rf}')
            print(f'Estimate / dominant true: {dr:.5e} ({1 / dr:.5e})')

            import matplotlib.pyplot as plt
            plt.show()

        with self.subTest(a=a, b=b, c=c, d=d, e=e, duration=duration, n=n,
                          fnoise=fnoise, t0=t0):
            self.assertLess(dr, rdom)
            self.assertGreater(dr, 1 / rdom)
            self.assertLess(rf, rmse)

    def test_single_on_triple(self):
        # Test single exponentials on single exponential data
        sot = self.single_on_triple
        self.r = np.random.default_rng(2)
        plot = False

        # Same direction
        sot(0, -6, -0.1, -3, -10, -2, -2, rdom=2.2, rmse=0.2, plot=plot)
        sot(0, -6, 0.1, -3, 10, -2, 2, rdom=1.002, rmse=650, plot=plot)
        sot(0, 3, -1, 3, -6, 2, -2, rdom=2, rmse=0.1, plot=plot)
        sot(0, 4, 0.2, 2.8, 10, 1.1, 20, rdom=1.001, rmse=6e6, plot=plot)

    def test_single_tau(self):

        a, b, c = 3, -1, 3
        t = np.linspace(0, 10, 10)
        v = a + b * np.exp(-t / c)
        r = expfit.fit_single_tau(t, v)
        self.assertAlmostEqual(r, 3, 3)

        # Negative infinity
        a, b, c = 1, 0, 3
        t = np.linspace(0, 10, 10)
        v = a + b * np.exp(-t / c)
        r = expfit.fit_single_tau(t, v)
        self.assertTrue(np.isinf(r))
        self.assertLess(r, 0)

    def test_single_with_peak_and_slope(self):
        # Remnant of "peak" at start of signal, plus slope at end

        a0, b0, c0 = 1, -2, -9
        n = 300
        x = np.linspace(0, 1, n)
        y = a0 * np.zeros(x.shape)
        y += b0 * np.exp(c0 * x)
        y += 0.8 * np.exp(-30 * x)
        y += -0.2 * x
        a, b, c = expfit.fit_single(x, y)
        self.assertAlmostEqual(a, a0, -1)
        self.assertAlmostEqual(b, b0, -1)
        self.assertAlmostEqual(c, c0, -1)
        self.assertLess(expfit.rmse_single(x, y, a, b, c), 0.1)

    def test_single_with_big_sine(self):
        # Sine wave causing both segment slopes to exceed the full signal slope

        a0, b0, c0 = 1, -2, -9
        n = 300
        x = np.linspace(0, 1, n)
        y = a0 * np.zeros(x.shape)
        y += b0 * np.exp(c0 * x)
        y += 0.1 * np.sin(10.2 * np.pi * x)
        a, b, c = expfit.fit_single(x, y)
        self.assertAlmostEqual(a, a0, -1)
        self.assertAlmostEqual(b, b0, 0)
        self.assertAlmostEqual(c, c0, 0)
        self.assertLess(expfit.rmse_single(x, y, a, b, c), 0.1)

    def test_single_ar1(self):
        # Test with AR1 noise

        a0, b0, c0 = 3, -4, -7
        x = np.linspace(0, 1, 300)
        y = a0 + b0 * np.exp(c0 * x)

        # Add AR1 noise
        # https://pints.readthedocs.io/en/stable/noise_generators.html
        rng = np.random.default_rng(5)
        rho, sigma = 0.9, 0.1
        s = sigma * np.sqrt(1 - rho**2)
        v = rng.normal(0, s, len(x))
        v[0] = rng.uniform()
        for t in range(1, len(x)):
            v[t] += rho * v[t - 1]
        y += v

        a, b, c = expfit.fit_single(x, y)
        self.assertAlmostEqual(a, a0, 1)
        self.assertAlmostEqual(b, b0, 0)
        self.assertAlmostEqual(c, c0, -1)
        self.assertLess(expfit.rmse_single(x, y, a, b, c), 0.1)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
