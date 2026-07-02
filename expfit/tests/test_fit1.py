#!/usr/bin/env python3
#
# Tests for single exponential fitting
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import unittest

import numpy as np

import expfit


class TestFit1(unittest.TestCase):
    """
    Tests fitting of single exponentials.
    """
    @classmethod
    def setUpClass(cls):
        # Create in each test and seed!
        cls.r = None

    def test_fit1_interface(self):

        # Test return type
        a0, b0, c0 = 3, -3, -4
        x = np.linspace(0, 1, 300)
        y = expfit.exp1(x, (a0, b0, c0))
        r = np.random.default_rng(5)
        y += r.normal(0, 1)
        p = expfit.fit1(x, y)
        self.assertIsInstance(p, expfit.ExponentialFit)

        # Test return type looks pretty
        r = expfit.ExponentialFit([1, 2, 3])
        self.assertEqual(str(r), '+1.00000e+00 +2.00000e+00 +3.00000e+00')

        # Test time series and x, y give same result
        q = expfit.fit1(expfit.TimeSeries(x, y))
        self.assertEqual(tuple(p), tuple(q))

    def test_fit1_edge_cases(self):
        # Test for a specific divide by zero case

        x = np.linspace(0, 1, 10)
        y = np.zeros(x.shape)   # Means scaling to unit square would div by 0
        self.assertRaises(expfit.NotExponentialError, expfit.fit1, x, y)

    def single_on_single(self, a, b, c, duration, n, fnoise=0.01, x0=0,
                         deltas=[], ratio=1, rmse=None, fails=False,
                         plot=False):
        """
        Fits an exponential to a signal containing an exponential, and returns
        the ratio ``RMSE(fit, noisy) / RMSE(true, noisy)``.

        Creates a signal ``a + b exp(c x)`` with ``n`` points from ``x0`` to
        ``x0 + duration``, and normally distributed noise with a variance of
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
        x = np.linspace(x0, x0 + duration, n)
        y = expfit.exp1(x, (a, b, c))
        s = max(fnoise * abs(y[0] - y[-1]), 1e-9)
        y += self.r.normal(0, s, size=n)

        try:
            af, bf, cf = expfit.fit1(
                x, y, plot=(a, b, c) if plot else False, opt_plot=plot)
        except expfit.FitFailedError as e:
            if not fails:  # pragma: no cover
                raise
            af, bf, cf = e.p
        finally:
            if plot:  # pragma: no cover
                import matplotlib.pyplot as plt
                plt.show()

        rt = expfit.rmse1(x, y, (a, b, c))
        rf = expfit.rmse1(x, y, (af, bf, cf))
        if plot:  # pragma: no cover
            print(f'RMSE true: {rt}')
            print(f'RMSE fit:  {rf}')
            print(f'ratio: {rf / rt}')

        with self.subTest(a=a, b=b, c=c, duration=duration, n=n, fnoise=fnoise,
                          x0=x0):
            if len(deltas) == 3:
                self.assertAlmostEqual(af, a, delta=deltas[0])
                self.assertAlmostEqual(bf, b, delta=deltas[1])
                self.assertAlmostEqual(cf, c, delta=deltas[2])
            if ratio is not None:
                self.assertLess(rf / rt, ratio)
            if rmse is not None:
                self.assertLess(rf, rmse)
            if len(deltas) != 3 and ratio is None and rmse is None:
                raise Exception('No test criteria set')  # pragma: no cover

    def test_fit1(self):
        # Test single exponentials on single exponential data
        sos = self.single_on_single
        self.r = np.random.default_rng(2)
        plot = False

        # Moderate
        sos(0, -1, 3, 2, 123, deltas=(1, .06, .03), plot=plot)
        sos(1, 3, -2, 5, 500, deltas=(.003, .003, .03), plot=plot)
        sos(5000, 3, 2, 5, 500, deltas=(100, .2, .01), plot=plot)
        sos(-1000, 10, 0.1, 50, 100, deltas=(3, 1, .002), plot=plot)
        sos(-600, 1, -1, 2, 3000, deltas=(.003, .002, .002), plot=plot)

        # Steep: These rely more on the guess than on the fitting
        # These would benefit from a de-steeper
        sos(4e5, -1, -30, 2, 300, plot=plot)
        sos(-999, 10, -10, 2, 1000, deltas=(.01, .1, .06), plot=plot)
        sos(3e5, -1, 15, 2, 500, fails=True, plot=plot)

        # Almost straight
        sos(0, -1, 0.3, 2, 123, deltas=(.1, .1, .02), plot=plot)
        sos(3, -1, 1, 2, 3000, deltas=(.01, .01, .01), plot=plot)
        sos(0, 1, -1e6, 1, 200, plot=plot)
        self.assertRaises(
            expfit.NotExponentialError,
            sos, 1, 2, -1e6, 1, 200, fnoise=0.2)

        # Both sides of zero
        sos(-2.5, 5, 0.5, 2, 50, deltas=(.6, .6, .04), plot=plot)

    def test_fit1_clean(self):
        sos = self.single_on_single
        self.r = np.random.default_rng(12)
        plot = False

        sos(0, -1, 3, 2, 123, fnoise=0,
            deltas=(1e-4, 1e-5, 5e-5), ratio=None, rmse=1e-5, plot=plot)
        sos(400, 2, 4, 2, 1000, fnoise=1e-3,
            deltas=(.03, .003, .0005), plot=plot)
        sos(7000, 3, -0.5, 5, 500, fnoise=1e-2,
            deltas=(0.007, .001, .003), plot=plot)

    def test_fit1_noisy(self):
        sos = self.single_on_single
        self.r = np.random.default_rng(12)
        plot = False

        # Noisy
        sos(4, 10, 3, 2, 100, fnoise=0.11, deltas=(40, 3, .2), plot=plot)
        sos(4, 10, 3, 2, 100, fnoise=0.3, deltas=(100, 4, .3), plot=plot)
        sos(51, -1, 0.4, 5, 200, fnoise=0.5, deltas=(.3, .3, .07), plot=plot)
        sos(-10, -2, 9, 2, 600, fnoise=1, deltas=(1e6, 5, 5), fails=True,
            plot=plot)

    def test_fit1_dense(self):
        # Test single exponentials on single exponential data
        sos = self.single_on_single
        self.r = np.random.default_rng(1)
        plot = False

        # Short
        sos(30, 2, 4, 2, 10, deltas=(15, .07, .01), plot=plot)
        sos(15, 3, 5, 1.5, 9, deltas=(5, .3, .07), plot=plot)
        sos(10, -3, -1e3, 5, 8, deltas=(.03, .01, 1000), plot=plot)
        sos(-2, 3, 0.05, 5, 6, deltas=(5, 3, .05), plot=plot)
        sos(-30, 10, -5, 0.2, 5, deltas=(1, 1, .5), plot=plot)
        sos(20, -10, 7, 2, 4, deltas=(4e5, 300, 3), plot=plot)

        # Dense
        sos(100, -2, 3, 2, 10000, deltas=(.1, .002, .001), plot=plot)
        sos(1000, 8, -0.12, 5, 100000, deltas=(.004, .003, 1e-4), plot=plot)

    def single_on_double(self, a, b, c, d, e, duration=1, n=100, fnoise=0.01,
                         x0=0, rmse=1, plot=False):
        """
        Fits a single exponential to a signal containing a double exponential.

        Criteria: ``rmse`` is the max rmse.
        """
        x = np.linspace(x0, x0 + duration, n)
        y = expfit.expd(x, (a, b, -1 / c, d, -1 / e))
        s = max(fnoise * abs(y[0] - y[-1]), 1e-9)
        y += self.r.normal(0, s, size=n)

        af, bf, cf = expfit.fit1(x, y, plot=plot)
        rf = expfit.rmse1(x, y, (af, bf, cf))

        # Dominant rate
        bdom, cdom = [(b, c), (d, e)][np.argmin(np.abs((c, e)))]

        if plot:  # pragma: no cover
            print(f'True dominant: {bdom:+.5e} {cdom:+.5e}')
            print(f'Full: a {a:+.5e}')
            print(f'      b {b:+.5e} c {c:+.5e}')
            print(f'      d {d:+.5e} e {e:+.5e}')
            print(f'RMSE fit: {rf}')

            import matplotlib.pyplot as plt
            plt.show()

        with self.subTest(a=a, b=b, c=c, d=d, e=e, duration=duration, n=n,
                          fnoise=fnoise, x0=x0):
            self.assertLess(rf, rmse)

    def test_fit1_on_double(self):
        # Test single exponentials on decaying single exponential data
        sod = self.single_on_double
        self.r = np.random.default_rng(2)
        plot = False

        # Same direction
        sod(0, -20, -10, -5, -5, rmse=0.5, plot=plot)
        sod(0, -8, -20, -2, -2, rmse=0.5, plot=plot)
        sod(0, -1, -10, -1, -3, rmse=0.05, plot=plot)
        sod(0, 1, -3, 1, -3.1, rmse=0.05, plot=plot)
        sod(0, 2, -30, 10, -0.1, rmse=0.3, plot=plot)
        sod(0, 2, -20, 20, -2, rmse=0.5, plot=plot)

    def single_on_triple(self, a, b, c, d, e, f, g, duration=1, n=100,
                         fnoise=0.01, x0=0, rmse=2, plot=False):
        """
        Fits a single exponential to a signal containing a double exponential.

        Criteria: ``rmse`` is the max rmse.
        """
        x = np.linspace(x0, x0 + duration, n)
        y = expfit.expd(x, (a, b, -1 / c, d, -1 / e, f, -1 / g))
        s = max(fnoise * abs(y[0] - y[-1]), 1e-9)
        y += self.r.normal(0, s, size=n)

        af, bf, cf = expfit.fit1(x, y, plot=plot)
        rf = expfit.rmse1(x, y, (af, bf, cf))

        # Dominant rate
        bdom, cdom = [(b, c), (d, e), (f, g)][np.argmin(np.abs((c, e, g)))]

        if plot:  # pragma: no cover
            print(f'True dominant: {bdom:+.5e} {cdom:+.5e}')
            print(f'Full: a {a:+.5e}')
            print(f'      b {b:+.5e} c {c:+.5e}')
            print(f'      d {d:+.5e} e {e:+.5e}')
            print(f'      f {f:+.5e} g {g:+.5e}')
            print(f'RMSE fit: {rf}')

            import matplotlib.pyplot as plt
            plt.show()

        with self.subTest(a=a, b=b, c=c, d=d, e=e, duration=duration, n=n,
                          fnoise=fnoise, x0=x0):
            self.assertLess(rf, rmse)

    def test_fit1_on_triple(self):
        # Test single fit on triple exponential data
        sot = self.single_on_triple
        self.r = np.random.default_rng(2)
        plot = False

        # Same direction
        sot(0, -100, -1, -100, -10, -50, -100, rmse=10, plot=plot)
        sot(0, 1, -2, 2, -10, 2, -50, rmse=0.3, plot=plot)
        sot(5, 5, -0.2, 5, -0.1, 5, -10, duration=5, rmse=1, plot=plot)

    def test_fit1_with_peak_and_slope(self):
        # Remnant of "peak" at start of signal, plus slope at end
        plot = False

        a0, b0, c0, d0, e0 = 1, -2, 0.1, 0.8, 0.03
        n = 300
        x = np.linspace(0, 1, n)
        y = expfit.expd(x, (a0, b0, c0, d0, e0))
        y += -0.2 * x
        a, b, c = expfit.fit1(x, y, plot=plot)
        if plot:  # pragma: no cover
            import matplotlib.pyplot as plt
            plt.show()
        self.assertAlmostEqual(a, a0, delta=0.3)
        self.assertAlmostEqual(b, b0, delta=1)
        self.assertAlmostEqual(c, -1 / c0, delta=1)
        self.assertLess(expfit.rmse1(x, y, (a, b, c)), 0.1)

    def test_fit1_with_ar1(self):
        # Test with AR1 noise
        plot = False

        a0, b0, c0 = 3, -4, -7
        x = np.linspace(0, 1, 300)
        y = expfit.exp1(x, (a0, b0, c0))

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

        a, b, c = expfit.fit1(x, y, plot=plot)
        if plot:  # pragma: no cover
            import matplotlib.pyplot as plt
            plt.show()
        self.assertAlmostEqual(a, a0, delta=0.1)
        self.assertAlmostEqual(b, b0, delta=0.2)
        self.assertAlmostEqual(c, c0, delta=1)
        self.assertLess(expfit.rmse1(x, y, (a, b, c)), 0.2)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
