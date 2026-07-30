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


def rmse1(x, y, a, b, c):
    return np.sqrt(np.sum((y - a - b * np.exp(c * x))**2) / len(x))


class TestLS(unittest.TestCase):
    """ Tests linear least squares """

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


class TestEst1(unittest.TestCase):
    """ Tests initial estimates for a single exponential. """
    @classmethod
    def setUpClass(cls):
        # Create in each test and seed!
        cls.r = None

    def test_est1_interface(self):
        # Test directly, check return type etc.

        p0 = 3, 5, 2
        x = np.linspace(0.5, 1.5, 100)
        y = expfit.exp1(x, p0)
        t = expfit.UnitSquaredSeries(x, y)
        r = expfit.est1(t)
        p = t.detransform(r)
        self.assertAlmostEqual(p[0], p0[0], delta=.1)
        self.assertAlmostEqual(p[1], p0[1], delta=.03)
        self.assertAlmostEqual(p[2], p0[2], delta=2e-3)

        # Result object, with nice str(), some properties, but no logs or zoom
        self.assertEqual(str(r), '-0.1565 0.1565 2')
        self.assertIsInstance(r.ls0, expfit.LeastSquaresFit)
        self.assertIsInstance(r.ls1, expfit.LeastSquaresFit)
        self.assertIsInstance(r.ls2, expfit.LeastSquaresFit)
        self.assertIsNone(r.region)
        self.assertIsNone(r.log1)
        self.assertIsNone(r.log2)

        # Result object with log info (but no zoom)
        r = expfit.est1(t, log=True)
        p = t.detransform(r)
        self.assertAlmostEqual(p[0], p0[0], delta=.1)
        self.assertAlmostEqual(p[1], p0[1], delta=.03)
        self.assertAlmostEqual(p[2], p0[2], delta=2e-3)
        self.assertIsInstance(r.ls0, expfit.LeastSquaresFit)
        self.assertIsInstance(r.ls1, expfit.LeastSquaresFit)
        self.assertIsInstance(r.ls2, expfit.LeastSquaresFit)
        self.assertIsNone(r.region)
        self.assertIsNotNone(r.log1)
        self.assertIsNotNone(r.log2)
        self.assertEqual(len(r.log1), 6)
        self.assertEqual(len(r.log2), 6)
        self.assertIsInstance(r.log1[0], expfit.LeastSquaresFit)
        self.assertIsInstance(r.log2[0], expfit.LeastSquaresFit)

        # With zoom too
        y = expfit.exp1(x, (3, 5, 10))
        t = expfit.UnitSquaredSeries(x, y)
        r = expfit.est1(t, log=False)
        self.assertIsNotNone(r.region)
        r = expfit.est1(t, log=True)
        self.assertIsNotNone(r.region)

        # Vets
        y = expfit.exp1(x, (3, 5, -1))
        self.assertRaisesRegex(
            ValueError, 'must have same length, got 100 and 99',
            expfit.est1, x, y[:-1])
        self.assertRaisesRegex(
            ValueError, 'At least 3', expfit.est1, [1, 2], [3, 4])

        # Extra info: No shrinking, data too small
        x = np.linspace(0, 1, 3)
        y = expfit.exp1(x, (3, 5, 2))
        t = expfit.UnitSquaredSeries(x, y)
        r = expfit.est1(t, log=True)
        self.assertEqual(len(r.log1), 1)
        self.assertEqual(len(r.log2), 1)

        # Equal slopes
        t = expfit.UnitSquaredSeries(x, x)
        self.assertRaises(
            expfit.InitialEstimateFailed, expfit.est1, t)

        # Contrived example with equal means but not equal slopes
        x = np.array([0, 1, 2, 3])
        y = np.array([0, 2, -1, 3])
        self.assertRaises(
            expfit.InitialEstimateFailed, expfit.est1, x, y)

    def est1(self, x, y, transform=True, plot=False):
        """
        Calls expfit.est1, after transforming to the unit square. Shows plot,
        if asked.
        """
        try:
            if transform:
                t = expfit.UnitSquaredSeries(x, y)
                r = expfit.est1(t, plot=plot)
            else:
                r = expfit.est1(x, y, plot=plot)
        finally:
            if plot:  # pragma: no cover
                import matplotlib.pyplot as plt
                plt.show()
        return t.detransform(r) if transform else r

    def test_est1(self):
        # Straightforward tests on a growing and decaying single exponential
        rng = np.random.default_rng(71)
        e = expfit.exp1
        plot = False

        a, b, c = 73, 1, 0.2
        n = 1003
        x = np.linspace(0, 6.7, n)
        y = e(x, (a, b, c)) + rng.normal(0, 0.05, n)
        p, q, r = self.est1(x, y, plot=plot)
        self.assertAlmostEqual(p, a, delta=.02)
        self.assertAlmostEqual(q, b, delta=.02)
        self.assertAlmostEqual(r, c, delta=1e-3)

        a, b, c = 1, 1e9, -3
        n = 88
        x = np.linspace(10, 11, n)
        y = e(x, (a, b, c)) + rng.normal(0, 1e-6, n)
        p, q, r = self.est1(x, y, plot=plot)
        self.assertAlmostEqual(p, a, delta=.01)
        self.assertAlmostEqual(r, c, delta=.02)

    def test_est1_clean(self):
        # Tests without noise

        e = expfit.exp1
        plot = False

        a, b, c = 8, 2, -3
        x = np.linspace(1.5, 2.5, 2000)
        p, q, r = self.est1(x, e(x, (a, b, c)), plot=plot)
        self.assertAlmostEqual(p, a, delta=2e-4)
        self.assertAlmostEqual(q, b, delta=.04)
        self.assertAlmostEqual(r, c, delta=.03)

        a, b, c = -1000, 5, 1
        x = np.linspace(0.3, 4, 200)
        p, q, r = self.est1(x, e(x, (a, b, c)), plot=plot)
        self.assertAlmostEqual(p, a, delta=1e-3)
        self.assertAlmostEqual(q, b, delta=1e-3)
        self.assertAlmostEqual(r, c, delta=1e-4)

        a, b, c = 200, 21, 10
        x = np.linspace(0, 0.5, 9)
        p, q, r = self.est1(x, e(x, (a, b, c)), plot=plot)
        self.assertAlmostEqual(p, a, delta=5)
        self.assertAlmostEqual(q, b, delta=5)
        self.assertAlmostEqual(r, c, delta=.5)

    def test_est1_straight(self):
        # Edge cases: straight and flat lines for est1
        # Most of these will (and should) only be caught in fit1(), which can
        # do statiscal tests based on an optimal fit. So the tests here are
        # for extreme cases.

        rng = np.random.default_rng(1)
        plot = False

        # Edge case: perfectly flat line, no noise
        x = np.linspace(0, 1, 10)
        y = 3 * np.ones(x.shape)
        self.assertRaises(
            expfit.InitialEstimateFailed, self.est1, x, y, plot=plot)

        # Straight line through origin, no noise
        x = np.linspace(0, 1, 10)
        y = 3 * x
        self.assertRaises(
            expfit.InitialEstimateFailed, self.est1, x, y, plot=plot)
        # Note: without the transform this isn't picked up

    def test_est1_steep(self):

        rng = np.random.default_rng(17)
        e = expfit.exp1
        plot = False

        # No zoom: Not steep enough
        a, b, c = 8, 2, 5
        x = np.linspace(0, 1, 20)
        p, q, r = self.est1(x, e(x, (a, b, c)), plot=plot)
        self.assertAlmostEqual(p, a, delta=.1)
        self.assertAlmostEqual(q, b, delta=.1)
        self.assertAlmostEqual(r, c, delta=.05)

        # No zoom: Too short
        a, b, c = 200, 21, 15
        x = np.linspace(0, 1, 40)
        p, q, r = self.est1(x, e(x, (a, b, c)), plot=plot)
        self.assertAlmostEqual(p, a, delta=10)
        self.assertAlmostEqual(q, b, delta=5)
        self.assertAlmostEqual(r, c, delta=.5)

        # Zoom
        a, b, c = 8, 2, 7
        x = np.linspace(0, 1, 500)
        p, q, r = self.est1(x, e(x, (a, b, c)), plot=plot)
        self.assertAlmostEqual(p, a, delta=.05)
        self.assertAlmostEqual(q, b, delta=1e-3)
        self.assertAlmostEqual(r, c, delta=1e-3)

        a, b, c = -1000, 5, -10
        x = np.linspace(0, 1, 200)
        p, q, r = self.est1(x, e(x, (a, b, c)), plot=plot)
        self.assertAlmostEqual(p, a, delta=1e-3)
        self.assertAlmostEqual(q, b, delta=5e-3)
        self.assertAlmostEqual(r, c, delta=5e-3)

        # With noise: Noise stops zoom from happening
        a, b, c = 8, 2, 7
        n = 500
        x = np.linspace(0, 1, n)
        y = e(x, (a, b, c)) + rng.normal(0, 50, n)
        p, q, r = self.est1(x, y, plot=plot)
        self.assertAlmostEqual(p, a, delta=10)
        self.assertAlmostEqual(q, b, delta=1)
        self.assertAlmostEqual(r, c, delta=.5)

        # With noise and zoom
        a, b, c = -5e4, -1e5, -20
        n = 900
        x = np.linspace(0, 1, n)
        y = e(x, (a, b, c)) + rng.normal(0, 9e2, n)
        p, q, r = self.est1(x, y, plot=plot)
        self.assertAlmostEqual(p, a, delta=300)
        self.assertAlmostEqual(q, b, delta=4000)
        self.assertAlmostEqual(r, c, delta=.2)
        self.assertLess(rmse1(x, y, p, q, r), 1000)

        a, b, c = 1e5, 1e5, 15
        n = 999
        x = np.linspace(0, 1, n)
        y = e(x, (a, b, c)) + rng.normal(0, 2e9, n)
        p, q, r = self.est1(x, y, plot=plot)
        self.assertAlmostEqual(p, a, delta=1e9)
        self.assertAlmostEqual(q, b, delta=1e5)
        self.assertAlmostEqual(r, c, delta=1)
        self.assertLess(rmse1(x, y, p, q, r), 1e10)

    def test_find_action(self):
        x = np.linspace(0, 1, 111)
        y = expfit.exp1(x, (8, 2, 7))

        r = expfit._est.find_action(x, y)
        self.assertEqual(r, (84, 111))

        x = np.linspace(0, 1, 50)
        y = expfit.exp1(x, (1, 1, 1))
        r = expfit._est.find_action(x, y)
        self.assertIsNone(r)


class TestEstd11(unittest.TestCase):
    """ Tests initial estimates for opposing decaying exponentials. """
    @classmethod
    def setUpClass(cls):
        # Create in each test and seed!
        cls.r = None

    def estd11(self, x, y, plot=False):
        """
        Calls estd11, after transforming to the unit square.
        Shows plot, if asked.
        """
        t = expfit.UnitSquaredSeries(x, y)
        try:
            r = expfit.estd11(t, plot=plot)
        finally:
            if plot:  # pragma: no cover
                import matplotlib.pyplot as plt
                plt.show()
        return t.detransform(r)

    def test_estd11(self):
        plot = False

        # No noise, down then up
        p = 8, -1, 0.2, 3, 0.15
        x = np.linspace(0.5, 1.5, 200)
        y = expfit.expd(x, p)
        q = self.estd11(x, y, plot=plot)
        q[2::2] = -1 / q[2::2]
        self.assertAlmostEqual(q[0], p[0], delta=0.01)
        self.assertAlmostEqual(q[1], p[1], delta=1)
        self.assertAlmostEqual(q[2], p[2], delta=1)
        self.assertAlmostEqual(q[3], p[3], delta=2)
        self.assertAlmostEqual(q[4], p[4], delta=.1)
        self.assertLess(expfit.rmsed(x, y, q), .001)

        # No noise, up then down (shallow)
        p = -3, 10, 0.17, -200, 0.1
        x = np.linspace(0.5, 1.5, 200)
        y = expfit.expd(x, p)
        q = self.estd11(x, y, plot=plot)
        q[2::2] = -1 / q[2::2]
        self.assertAlmostEqual(q[0], p[0], delta=0.02)
        self.assertAlmostEqual(q[1], p[1], delta=10)
        self.assertAlmostEqual(q[2], p[2], delta=.6)
        self.assertAlmostEqual(q[3], p[3], delta=150)
        self.assertAlmostEqual(q[4], p[4], delta=.03)
        self.assertLess(expfit.rmsed(x, y, q), 0.02)

        # Down-up with noise. Exact noise matters
        p0 = 1, -4, .5, 6, .2
        x = np.linspace(0, 2, 500, endpoint=False)
        r = np.random.default_rng(38)
        y = expfit.expd(x, p0) + r.normal(0, 0.3, size=x.shape)
        self.estd11(x, y, plot=plot)
        r = np.random.default_rng(1)
        y = expfit.expd(x, p0) + r.normal(0, 0.3, size=x.shape)
        self.estd11(x, y, plot=plot)
        r = np.random.default_rng(17)
        y = expfit.expd(x, p0) + r.normal(0, 0.3, size=x.shape)
        self.estd11(x, y, plot=plot)

        # Test size check
        p = -3, 10, 0.17
        x = np.linspace(0, 1, 5)
        y = expfit.expd(x, p)
        self.assertRaisesRegex(
            ValueError, 'At least 10 points', expfit.estd11, x, y)

        # Test not opposing
        e = self.estd11
        x = np.linspace(0, 1, 50)

        # Flat line, no noise
        y = np.ones(x.shape)
        self.assertRaisesRegex(
            expfit.NotOpposingError, 'Second segment', e, x, y, plot=plot)

        # Flat line, som noise - as it happens, (with seed 1), the first
        # segment isn't recognisably exponential
        r = np.random.default_rng(1)
        y = np.ones(x.shape) + r.normal(0, 0.1, size=x.shape)
        self.assertRaisesRegex(
            expfit.NotOpposingError, 'First segment', e, x, y, plot=plot)
        # Different seed: now the second segment looks expanding
        r = np.random.default_rng(2)
        y = np.ones(x.shape) + r.normal(0, 0.1, size=x.shape)
        self.assertRaisesRegex(
            expfit.NotOpposingError, 'not both decaying', e, x, y, plot=plot)
        # Different seed: now it actually looks about right
        r = np.random.default_rng(5)
        y = np.ones(x.shape) + r.normal(0, 0.1, size=x.shape)
        self.estd11(x, y, plot=True)

        # TODO: Make that fail?

        #return
        y = expfit.exp1(x, (1, 1, -1))
        self.assertRaisesRegex(
            expfit.NotOpposingError, 'lalala', e, x, y, plot=True)
        y = expfit.exp1(x, (1, 1, 1))
        self.assertRaisesRegex(
            expfit.NotOpposingError, 'lalala', e, x, y, plot=True)
        # Looks like one exponential
        y = expfit.expd(x, (1, 1, 1, 1, -1))
        self.assertRaisesRegex(
            expfit.NotDecayingError, 'lalala', e, x, y, plot=True)

        # Not opposing, harder
        r = np.random.default_rng(1)
        p0 = 5, 5, 5, 5, 1, 5, .1
        x = np.linspace(0, 5, 300, endpoint=False)
        y = expfit.expd(x, p0) + r.normal(0, 0.1, size=x.shape)
        self.assertRaises(
            expfit.NotOpposingError, self.estd11, x, y, plot=plot)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
