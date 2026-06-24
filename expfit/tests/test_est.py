#!/usr/bin/env python3
#
# Tests for the initial estimates
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import unittest

import numpy as np

import expfit


class TestEstimates(unittest.TestCase):
    """
    Tests initial estimates.
    """
    @classmethod
    def setUpClass(cls):
        # Create in each test and seed!
        cls.r = None

    def test_least_squares(self):
        # Test linear least squares

        x = np.array([-5, -2, 0, 0.1, 3, 8, 13])
        y = 4 + 13 * x
        ls = expfit.LeastSquaresFit(x, y)
        self.assertEqual(ls.offset, 4)
        self.assertEqual(ls.slope, 13)
        self.assertEqual(ls.mu_x, np.mean(x))
        self.assertEqual(ls.mu_y, np.mean(y))

        # Test string representation
        self.assertEqual(str(ls), 'mu (2.44, 35.8), 4.0 + 13.0 x')
        self.assertEqual(repr(ls), '<expfit.LeastSquaresFit(4.0+13.0x)>')

        # Test array checks
        x = np.array([[1, 2, 3]])
        self.assertRaisesRegex(
            ValueError, 'must be 1-dimensional', expfit.LeastSquaresFit, x, x)
        x, y = [1, 2, 3], [4, 5]
        self.assertRaisesRegex(
            ValueError, 'must have same length', expfit.LeastSquaresFit, x, y)
        self.assertRaisesRegex(
            ValueError, 't least 2 points', expfit.LeastSquaresFit, [1], [2])

    def test_estimate_initial_basics(self):
        # Test directly, check return type etc.

        p0 = 3, 5, 1
        x = np.linspace(0.5, 1.5, 100)
        y = expfit.exp(x, p0)
        tv = expfit.UnitSquareTransformedTimeSeries(x, y)
        r = expfit.estimate_initial_single(tv)
        p = tv.detransform(r)
        self.assertAlmostEqual(p[0], p0[0], delta=1e-3)
        self.assertAlmostEqual(p[1], p0[1], delta=1e-3)
        self.assertAlmostEqual(p[2], p0[2], delta=1e-3)

        # Result object, with nice str() but no extended info
        self.assertEqual(str(r), '-0.582 1.582 1')
        self.assertIsNone(r.log1)
        self.assertIsNone(r.log2)
        self.assertIsNone(r.region)

        # Result object with full info (but no zoom)
        r = expfit.estimate_initial_single(tv, full=True)
        p = tv.detransform(r)
        self.assertAlmostEqual(p[0], p0[0], delta=1e-3)
        self.assertAlmostEqual(p[1], p0[1], delta=1e-3)
        self.assertAlmostEqual(p[2], p0[2], delta=1e-3)
        self.assertIsNone(r.region)
        self.assertIsNotNone(r.log1)
        self.assertIsNotNone(r.log2)
        self.assertGreater(len(r.log1), 0)
        self.assertGreater(len(r.log2), 0)
        self.assertEqual(len(r.log1[0]), 2)
        self.assertEqual(len(r.log2[0]), 2)
        self.assertIsInstance(r.log1[0][0], expfit.LeastSquaresFit)
        self.assertIsInstance(r.log2[0][0], expfit.LeastSquaresFit)

        # With zoom too
        y = expfit.exp(x, (3, 5, -0.1))
        tv = expfit.UnitSquareTransformedTimeSeries(x, y)
        r = expfit.estimate_initial_single(tv, full=True)
        self.assertIsNotNone(r.region)

        # Vets
        y = expfit.exp(x, (3, 5, 1))
        self.assertRaisesRegex(
            ValueError, 'must have same length, got 100 and 99',
            expfit.estimate_initial_single, x, y[:-1])
        self.assertRaisesRegex(
            ValueError, 'At least 3', expfit.estimate_initial_single,
            [1, 2], [3, 4])

        # Extra info: No shrinking, data too small
        x = np.linspace(0, 1, 3)
        y = expfit.exp(x, (3, 5, -0.5))
        tv = expfit.UnitSquareTransformedTimeSeries(x, y)
        r = expfit.estimate_initial_single(tv, full=True)
        self.assertEqual(len(r.log1), 1)
        self.assertEqual(len(r.log2), 1)
        self.assertEqual(r.log1[0][1], 'Initial segment at minimum size')
        self.assertEqual(r.log2[0][1], 'Initial segment at minimum size')

        # Equal slopes
        tv = expfit.UnitSquareTransformedTimeSeries(x, x)
        self.assertRaisesRegex(
            expfit.NotExponentialError, 'Equal slopes',
            expfit.estimate_initial_single, tv)
        self.assertRaisesRegex(
            expfit.NotExponentialError, 'Equal slopes',
            expfit.estimate_initial_single, tv, full=True)

        # Contrived example with equal means but not equal slopes
        x = np.array([0, 1, 2, 3])
        y = np.array([0, 2, -1, 3])
        self.assertRaisesRegex(
            expfit.NotExponentialError, 'Equal means',
            expfit.estimate_initial_single, x, y)

    def estimate_initial(self, x, y, transform=True, plot=False):
        """
        Calls estimate_initial_single, after transforming to the unit square.
        Shows plot, if asked.
        """
        if transform:
            tv = expfit.UnitSquareTransformedTimeSeries(x, y)
            ret = expfit.estimate_initial_single(tv, plot=plot)
        else:
            ret = expfit.estimate_initial_single(x, y, plot=plot)
        if plot:  # pragma: no cover
            import matplotlib.pyplot as plt
            plt.show()
        return tv.detransform(*ret) if transform else ret

    def test_estimate_initial(self):

        rng = np.random.default_rng(71)
        e = expfit.exp
        plot = False

        # Noise free
        a, b, t = 8, 2, -3
        x = np.linspace(1.5, 2.5, 2000)
        p, q, r = self.estimate_initial(x, e(x, (a, b, t)), plot=plot)
        self.assertAlmostEqual(p, a, delta=1e-8)
        self.assertAlmostEqual(q, b, delta=2e-7)
        self.assertAlmostEqual(r, t, delta=1e-7)

        a, b, t = -1000, 5, 3
        x = np.linspace(0.3, 4, 200)
        p, q, r = self.estimate_initial(x, e(x, (a, b, t)), plot=plot)
        self.assertAlmostEqual(p, a, delta=1e-10)
        self.assertAlmostEqual(q, b, delta=2e-4)
        self.assertAlmostEqual(r, t, delta=2e-4)

        a, b, t = 200, 21, 1.5
        x = np.linspace(0, 0.5, 9)
        p, q, r = self.estimate_initial(x, e(x, (a, b, t)), plot=plot)
        self.assertAlmostEqual(p, a, delta=1e-11)
        self.assertAlmostEqual(q, b, delta=0.1)
        self.assertAlmostEqual(r, t, delta=2e-3)

        # With noise
        a, b, t = 73, 1, -5
        n = 1003
        x = np.linspace(0, 6.7, n)
        y = e(x, (a, b, t)) + rng.normal(0, 0.05, n)
        p, q, r = self.estimate_initial(x, y, plot=plot)
        self.assertAlmostEqual(p, a, delta=0.4)
        self.assertAlmostEqual(q, b, delta=0.5)
        self.assertAlmostEqual(r, t, delta=2)

        a, b, t = 1, 1e13, 0.4
        n = 88
        x = np.linspace(10, 11, n)
        y = e(x, (a, b, t)) + rng.normal(0, 0.02, n)
        p, q, r = self.estimate_initial(x, y, plot=plot)
        self.assertAlmostEqual(p, a, delta=0.5)
        self.assertAlmostEqual(q, b, delta=8e12)
        self.assertAlmostEqual(r, t, delta=0.1)
        self.assertLess(expfit.rmse(x, y, (p, q, r)), 0.2)

    def test_estimate_initial_straight(self):
        # Edge cases: straight and flat lines for estimate_initial_single

        rng = np.random.default_rng(1)
        plot = False

        # Edge case: perfectly flat line, no noise
        x = np.linspace(0, 1, 10)
        y = 3 * np.ones(x.shape)
        self.assertRaisesRegex(
            expfit.NotExponentialError, 'Equal slopes',
            self.estimate_initial, x, y)

        # Flat line with noise
        x = np.linspace(0, 1, 3000)
        y = 3 * np.ones(x.shape) + rng.normal(0, 1e-9, x.shape)
        p, q, r = self.estimate_initial(x, y, plot=plot)
        self.assertAlmostEqual(p, 3, delta=1e-10)
        self.assertAlmostEqual(q, 0, delta=1e-10)

        # Straight line through origin, no noise
        # Note: the transform amplifies the numerical noise here, causing the
        # RMSE to be non-zero for the transformed case.
        rmse = expfit.rmse
        x = np.linspace(0, 1, 10)
        y = 3 * x
        p, q, r = self.estimate_initial(x, y, plot=plot)
        self.assertLess(rmse(x, y, (p, q, r)), 0.6)
        self.assertEqual(p, -q)
        p, q, r = self.estimate_initial(x, y, transform=False, plot=plot)
        self.assertEqual(rmse(x, y, (p, q, r)), 0)
        self.assertEqual(p, -q)

        # Straight line through origin, with noise
        x = np.linspace(0, 1, 99)
        y = 3 * x + rng.normal(0, 0.1, x.shape)
        p, q, r = self.estimate_initial(x, y, plot=plot)
        self.assertAlmostEqual(p, -q, delta=0.1)
        self.assertLess(rmse(x, y, (p, q, r)), 0.2)

        # Straight line with offset and noise
        x = np.linspace(0, 1, 99)
        y = 4 + 2 * x + rng.normal(0, 0.1, x.shape)
        p, q, r = self.estimate_initial(x, y, plot=plot)
        self.assertAlmostEqual(p + q, 4, delta=0.5)
        self.assertLess(rmse(x, y, (p, q, r)), 0.2)

        # Almost flat with noise
        a, b, t = -51, -7.2, 1e3
        n = 900
        x = np.linspace(1e-3, 7e-3, n)
        y = expfit.exp(x, (a, b, t)) + rng.normal(0, 100, n)
        self.assertRaisesRegex(
            expfit.NotExponentialError, 'Flat line',
            self.estimate_initial, x, y)

        # This failed when the ZoomTransform was variance-based
        # Specific case: needs this seed
        rng = np.random.default_rng(2)
        x = np.linspace(0, 1, 200)
        y = 3 * np.zeros(x.shape)
        y += rng.normal(0, 1, x.shape)
        self.assertRaisesRegex(
            expfit.NotExponentialError, 'Flat line',
            self.estimate_initial, x, y)

    def test_estimate_initial_steep(self):

        rng = np.random.default_rng(17)
        e = expfit.exp
        plot = False

        # No zoom: Not steep enough
        a, b, t = 8, 2, -0.2
        x = np.linspace(0, 1, 20)
        p, q, r = self.estimate_initial(x, e(x, (a, b, t)), plot=plot)
        self.assertAlmostEqual(p, a, delta=1e-8)
        self.assertAlmostEqual(q, b, delta=1)
        self.assertAlmostEqual(r, t, delta=.01)

        # No zoom: Too short
        a, b, t = 200, 21, -.07
        x = np.linspace(0, 1, 40)
        p, q, r = self.estimate_initial(x, e(x, (a, b, t)), plot=plot)
        self.assertAlmostEqual(p, a, delta=1e-11)
        self.assertAlmostEqual(q, b, delta=500)
        self.assertAlmostEqual(r, t, delta=.03)

        # Zoom
        a, b, t = 8, 2, -.14
        x = np.linspace(0, 1, 500)
        p, q, r = self.estimate_initial(x, e(x, (a, b, t)), plot=plot)
        self.assertAlmostEqual(p, a, delta=1e-8)
        self.assertAlmostEqual(q, b, delta=1e-2)
        self.assertAlmostEqual(r, t, delta=1e-4)

        a, b, t = -1000, 5, 0.1
        x = np.linspace(0, 1, 200)
        p, q, r = self.estimate_initial(x, e(x, (a, b, t)), plot=plot)
        self.assertAlmostEqual(p, a, delta=1e-10)
        self.assertAlmostEqual(q, b, delta=1e-3)
        self.assertAlmostEqual(r, t, delta=1e-3)

        # With noise: Noise stops zoom from happening
        a, b, t = 8, 2, -.14
        n = 500
        x = np.linspace(0, 1, n)
        y = e(x, (a, b, t)) + rng.normal(0, 50, n)
        p, q, r = self.estimate_initial(x, y, plot=plot)
        self.assertAlmostEqual(p, a, delta=20)
        self.assertAlmostEqual(q, b, delta=2)
        self.assertAlmostEqual(r, t, delta=.02)

        # With noise and zoom
        a, b, t = -5e4, -1e5, .05
        n = 900
        x = np.linspace(0, 1, n)
        y = e(x, (a, b, t)) + rng.normal(0, 9e2, n)
        p, q, r = self.estimate_initial(x, y, plot=plot)
        self.assertAlmostEqual(p, a, delta=5e2)
        self.assertAlmostEqual(q, b, delta=8e3)
        self.assertAlmostEqual(r, t, delta=.04)
        self.assertLess(expfit.rmse(x, y, (p, q, r)), 4e3)

        a, b, t = 1e5, 1e5, -.07
        n = 999
        x = np.linspace(0, 1, n)
        y = e(x, (a, b, t)) + rng.normal(0, 2e9, n)
        p, q, r = self.estimate_initial(x, y, plot=plot)
        self.assertAlmostEqual(p, a, delta=1e10)
        self.assertAlmostEqual(q, b, delta=1e6)
        self.assertAlmostEqual(r, t, delta=.03)
        self.assertLess(expfit.rmse(x, y, (p, q, r)), 1e10)

    def estimate_initial_opposing(self, x, y, plot=False):
        """
        Calls estimate_initial_single, after transforming to the unit square.
        Shows plot, if asked.
        """
        tr = expfit.UnitSquareTransform(x, y)
        x, y = tr.x, tr.y
        ret = expfit.estimate_initial_opposing(x, y, plot=plot)
        if plot:  # pragma: no cover
            import matplotlib.pyplot as plt
            plt.show()
        return tr.detransform(*ret)

    '''
    def test_estimate_initial_opposing(self):
        e = expfit.expc
        plot = False

        p = 8, -1, -5, 3, -7
        x = np.linspace(0.5, 1.5, 200)
        q = self.estimate_initial_opposing(x, e(x, p), plot=plot)
        self.assertAlmostEqual(q[0], 8, delta=0.001)
        self.assertAlmostEqual(q[1], -1, delta=2)
        self.assertAlmostEqual(q[2], -5, delta=3)
        self.assertAlmostEqual(q[3], 3, delta=2)
        self.assertAlmostEqual(q[4], -7, delta=4)

        p = -3, 10, -6, -200, -10
        x = np.linspace(0.5, 1.5, 200)
        q = self.estimate_initial_opposing(x, e(x, p), plot=plot)
        self.assertAlmostEqual(q[0], -3, delta=0.005)
        self.assertAlmostEqual(q[1], 10, delta=10)
        self.assertAlmostEqual(q[2], -6, delta=3)
        self.assertAlmostEqual(q[3], -200, delta=120)
        self.assertAlmostEqual(q[4], -10, delta=2)

        # Test vetting can be disabled
        x = np.linspace(0, 1, 10)
        expfit.estimate_initial_opposing(x, e(x, p))
        self.assertRaisesRegex(
            ValueError, 'Both arrays in series must have same length',
            expfit.estimate_initial_opposing, x, e(x[1:], p))
        self.assertRaisesRegex(
            ValueError, 'operands could not be broadcast',
            expfit.estimate_initial_opposing, x, e(x[1:], p), vet=False)

        # Test size check
        x = np.linspace(0, 1, 5)
        self.assertRaisesRegex(
            ValueError, 'At least 10 points',
            expfit.estimate_initial_opposing, x, e(x, p))
    '''

    '''
    def test_find_action(self):
        x = np.linspace(0, 1, 111)
        y = expfit.exp(x, (8, 2, -1 / 7))

        ij = expfit.

        tr = expfit.ZoomTransform(x, y)
        a, b, c = 1, 2, 3
        p, q, r = tr.transform(a, b, c)
        self.assertEqual(p, a)
        self.assertAlmostEqual(q, 19.76783796)
        self.assertAlmostEqual(r, 0.7090909091)
        u, v, w = tr.detransform(p, q, r)
        self.assertEqual(u, a)
        self.assertAlmostEqual(v, b)
        self.assertEqual(w, c)
        t, v = np.array([0.5, 0.6, 0.7]), np.array([0.1, 0.2, 0.3])
        x, y = tr.detransform_series(t, v)
        self.assertIs(v, y)
        self.assertEqual(len(x), 3)
        self.assertAlmostEqual(x[0], 0.88181818)
        self.assertAlmostEqual(x[1], 0.90545455)
        self.assertAlmostEqual(x[2], 0.92909091)

        x = np.linspace(0, 1, 50)
        y = expfit.exp(x, (1, 1, -1))
        tr = expfit.ZoomTransform(x, y)
        a, b, c = 1, 2, 3
        p, q, r = tr.transform(a, b, c)
        self.assertEqual(p, a)
        self.assertEqual(q, b)
        self.assertEqual(r, c)
        u, v, w = tr.detransform(p, q, r)
        self.assertEqual(u, a)
        self.assertEqual(v, b)
        self.assertEqual(w, c)
        t, v = np.array([0.5, 0.6, 0.7]), np.array([0.1, 0.2, 0.3])
        x, y = tr.detransform_series(t, v)
        self.assertIs(t, x)
        self.assertIs(v, y)
    '''

    '''
    def test_estimate_noise_level(self):
        # Test noise level estimates

        rng = np.random.default_rng(18)
        f = expfit.exp # CHANGED
        plot = False

        # Very straight line
        s = 0.5
        x = np.linspace(1.5, 2.5, 2000)
        y = f(x, (8, 2, -3)) + rng.normal(0, s, x.shape)
        e = expfit.estimate_noise_level(x, y, plot=plot)
        self.assertAlmostEqual(e / s, 1, delta=0.05)

        # Quite a strong exponential, low loise
        s = 0.03
        x = np.linspace(0.3, 4, 200)
        y = f(x, (-1000, 5, 0.5)) + rng.normal(0, s, x.shape)
        e = expfit.estimate_noise_level(x, y, plot=plot)
        self.assertAlmostEqual(e / s, 1, delta=0.2)

        # Not enough data, not enough noise
        s = 0.05
        x = np.linspace(0, 0.5, 20)
        y = f(x, (200, 21, 1.4)) + rng.normal(0, s, x.shape)
        e = expfit.estimate_noise_level(x, y, plot=plot)
        self.assertAlmostEqual(e / s, 1, delta=0.6)

        # Lots of data, strong noise
        s = 0.1
        x = np.linspace(0, 6.7, 1003)
        y = f(x, (73, 1, -5.55)) + rng.normal(0, s, x.shape)
        e = expfit.estimate_noise_level(x, y, plot=plot)
        self.assertAlmostEqual(e / s, 1, delta=0.1)

        # Strong noise, but massive exponential
        s = 10
        x = np.linspace(1e-3, 7e-3, 900)
        y = f(x, (-51, -7.2, -1e-3)) + rng.normal(0, s, x.shape)
        e = expfit.estimate_noise_level(x, y, plot=plot)
        self.assertAlmostEqual(e / s, 1, delta=0.1)

        # Down-then-up exponential
        s = 0.10
        x = np.linspace(0, 2, 800)
        y = f(x, (8, 10, -15, -15, -5)) + rng.normal(0, s, x.shape)
        e = expfit.estimate_noise_level(x, y, plot=plot)
        self.assertAlmostEqual(e / s, 1, delta=0.02)

        # Same but gentler
        s = 0.05
        x = np.linspace(0, 0.4, 800)
        y = f(x, (8, 10, -15, -15, -5)) + rng.normal(0, s, x.shape)
        e = expfit.estimate_noise_level(x, y, plot=plot)
        self.assertAlmostEqual(e / s, 1, delta=0.1)

        # Difficult one for fit1
        s = 1
        x = np.linspace(0, 5, 400)
        y = f(x, (5, 10, -.2, 5, -1, 5, -3.5, 10, -20))
        y += rng.normal(0, s, x.shape)
        e = expfit.estimate_noise_level(x, y, plot=True)
        #self.assertAlmostEqual(e / s, 1, delta=0.1)

        import matplotlib.pyplot as plt
        plt.show()

        # Vets, but can be disabled
        x = np.linspace(0, 1, 50)
        y = f(x[1:], (1, 2, -3))
        self.assertRaisesRegex(
            ValueError, 'must have same length, got 50 and 49',
            expfit.estimate_noise_level, x, y)
        # No error:
        expfit.estimate_noise_level(x, y, vet=False)
    '''


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
