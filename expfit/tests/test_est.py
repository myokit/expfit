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

    def estimate_initial(self, x, y, plot=False, transform=True):
        """
        Calls estimate_initial_single, after transforming to the unit square.
        Shows plot, if asked.
        """
        if transform:
            tr = expfit.UnitSquareTransform(x, y)
            x, y = tr.x, tr.y
        ret = expfit.estimate_initial_single(x, y, plot=plot)
        if plot:  # pragma: no cover
            import matplotlib.pyplot as plt
            plt.show()
        return tr.detransform(*ret) if transform else ret

    def test_estimate_initial(self):

        rng = np.random.default_rng(71)
        f = expfit.exp
        plot = False

        # Noise free
        a, b, c = 8, 2, 0.3
        x = np.linspace(1.5, 2.5, 2000)
        p, q, r = self.estimate_initial(x, f(x, (a, b, c)), plot=plot)
        self.assertAlmostEqual(p, a, delta=1e-9)
        self.assertAlmostEqual(q, b, delta=1e-7)
        self.assertAlmostEqual(r, c, delta=1e-8)

        a, b, c = -1000, 5, -0.3
        x = np.linspace(0.3, 4, 200)
        p, q, r = self.estimate_initial(x, f(x, (a, b, c)), plot=plot)
        self.assertAlmostEqual(p, a, delta=1e-10)
        self.assertAlmostEqual(q, b, delta=2e-4)
        self.assertAlmostEqual(r, c, delta=1e-5)

        a, b, c = 200, 21, -0.7
        x = np.linspace(0, 0.5, 9)
        p, q, r = self.estimate_initial(x, f(x, (a, b, c)), plot=plot)
        self.assertAlmostEqual(p, a, delta=1e-11)
        self.assertAlmostEqual(q, b, delta=0.1)
        self.assertAlmostEqual(r, c, delta=1e-3)

        # With noise
        a, b, c = 73, 1, 0.18
        n = 1003
        x = np.linspace(0, 6.7, n)
        y = f(x, (a, b, c)) + rng.normal(0, 0.05, n)
        p, q, r = self.estimate_initial(x, y, plot=plot)
        self.assertAlmostEqual(p, a, delta=0.2)
        self.assertAlmostEqual(q, b, delta=0.2)
        self.assertAlmostEqual(r, c, delta=0.02)

        a, b, c = -51, -7.2, 1000
        n = 900
        x = np.linspace(1e-3, 7e-3, n)
        y = f(x, (a, b, c)) + rng.normal(0, 100, n)
        p, q, r = self.estimate_initial(x, y, plot=plot)
        self.assertAlmostEqual(p, a, delta=6)
        self.assertAlmostEqual(q, b, delta=2)
        self.assertAlmostEqual(r, c, delta=40)

        a, b, c = 1, 1e13, -3
        n = 88
        x = np.linspace(10, 11, n)
        y = f(x, (a, b, c)) + rng.normal(0, 0.02, n)
        p, q, r = self.estimate_initial(x, y, plot=plot)
        self.assertAlmostEqual(p, a, delta=0.02)
        self.assertAlmostEqual(q, b, delta=8e12)
        self.assertAlmostEqual(r, c, delta=0.1)
        self.assertLess(expfit.rmse(x, y, (p, q, r)), 0.1)

        # Contrived example with equal means (but not equal slope)
        x = np.array([0, 1, 2, 3])
        y = np.array([0, 2, -1, 3])
        p, q, r = self.estimate_initial(x, y, plot=plot)
        self.assertEqual(p, 1)
        self.assertEqual(q, 0)
        self.assertEqual(r, 0)

        # Vets, but can be disabled
        a, b, c = 3, 5, -0.7
        x = np.linspace(0.5, 1.5, 100)
        y = f(x, (a, b, c))
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

        rng = np.random.default_rng(1)
        plot = False

        # Edge case: perfectly flat line, no noise
        x = np.linspace(0, 1, 10)
        y = 3 * np.ones(x.shape)
        p, q, r = self.estimate_initial(x, y, plot=plot)
        self.assertEqual((p, q, r), (3, 0, 0))

        # Flat line with noise
        x = np.linspace(0, 1, 3000)
        y = 3 * np.ones(x.shape) + rng.normal(0, 1e-9, x.shape)
        p, q, r = self.estimate_initial(x, y, plot=plot)
        self.assertAlmostEqual(p, 3, delta=1e-10)
        self.assertAlmostEqual(q, 0, delta=1e-10)

        # Straight line through origin, no noise
        # Note: the transform amplifies the numerical noise here, causing the
        # RMSE to be non-zero for the transformed case.
        x = np.linspace(0, 1, 10)
        y = 3 * x
        p, q, r = self.estimate_initial(x, y, plot=plot)
        self.assertLess(expfit.rmse(x, y, (p, q, r)), 0.6)
        self.assertEqual(p, -q)
        p, q, r = self.estimate_initial(x, y, transform=False, plot=plot)
        self.assertEqual(expfit.rmse(x, y, (p, q, r)), 0)
        self.assertEqual(p, -q)

        # Straight line through origin, with noise
        x = np.linspace(0, 1, 99)
        y = 3 * x + rng.normal(0, 0.1, x.shape)
        p, q, r = self.estimate_initial(x, y, plot=plot)
        self.assertAlmostEqual(p, -q, delta=0.1)
        self.assertLess(expfit.rmse(x, y, (p, q, r)), 0.2)

        # Straight line with offset and noise
        x = np.linspace(0, 1, 99)
        y = 4 + 2 * x + rng.normal(0, 0.1, x.shape)
        p, q, r = self.estimate_initial(x, y, plot=plot)
        self.assertAlmostEqual(p + q, 4, delta=0.5)
        self.assertLess(expfit.rmse(x, y, (p, q, r)), 0.2)

        # Specific case: needs this seed
        # This failed when the ZoomTransform was variance-based
        rng = np.random.default_rng(2)
        x = np.linspace(0, 1, 200)
        y = 3 * np.zeros(x.shape)
        y += rng.normal(0, 1, x.shape)
        p, q, r = self.estimate_initial(x, y, plot=plot)
        self.assertAlmostEqual(p + q, 0, delta=0.01)
        self.assertLess(expfit.rmse(x, y, (p, q, r)), 1)

    def test_estimate_initial_steep(self):

        rng = np.random.default_rng(17)
        f = expfit.exp
        plot = False

        # No zoom: Not steep enough
        a, b, c = 8, 2, 6
        x = np.linspace(0, 1, 2000)
        p, q, r = self.estimate_initial(x, f(x, (a, b, c)), plot=plot)
        self.assertAlmostEqual(p, a, delta=1e-9)
        self.assertAlmostEqual(q, b, delta=1)
        self.assertAlmostEqual(r, c, delta=1)

        # No zoom: Too short
        a, b, c = 200, 21, 15
        x = np.linspace(0, 1, 40)
        p, q, r = self.estimate_initial(x, f(x, (a, b, c)), plot=plot)
        self.assertAlmostEqual(p, a, delta=1e-9)
        self.assertAlmostEqual(q, b, delta=500)
        self.assertAlmostEqual(r, c, delta=5)

        # Zoom
        a, b, c = 8, 2, 7
        x = np.linspace(0, 1, 500)
        p, q, r = self.estimate_initial(x, f(x, (a, b, c)), plot=plot)
        self.assertAlmostEqual(p, a, delta=1e-9)
        self.assertAlmostEqual(q, b, delta=1e-2)
        self.assertAlmostEqual(r, c, delta=1e-3)

        a, b, c = -1000, 5, -10
        x = np.linspace(0, 1, 200)
        p, q, r = self.estimate_initial(x, f(x, (a, b, c)), plot=plot)
        self.assertAlmostEqual(p, a, delta=1e-10)
        self.assertAlmostEqual(q, b, delta=5e-4)
        self.assertAlmostEqual(r, c, delta=5e-2)

        # With noise: Noise stops zoom from happening
        a, b, c = 8, 2, 7
        n = 500
        x = np.linspace(0, 1, n)
        y = f(x, (a, b, c)) + rng.normal(0, 50, n)
        p, q, r = self.estimate_initial(x, y, plot=plot)
        self.assertAlmostEqual(p, a, delta=20)
        self.assertAlmostEqual(q, b, delta=2)
        self.assertAlmostEqual(r, c, delta=1)

        # With noise and zoom
        a, b, c = -5e4, -1e5, -20
        n = 900
        x = np.linspace(0, 1, n)
        y = f(x, (a, b, c)) + rng.normal(0, 9e2, n)
        p, q, r = self.estimate_initial(x, y, plot=plot)
        self.assertAlmostEqual(p, a, delta=5e2)
        self.assertAlmostEqual(q, b, delta=8e3)
        self.assertAlmostEqual(r, c, delta=10)
        self.assertLess(expfit.rmse(x, y, (p, q, r)), 4e3)

        a, b, c = 1e5, 1e5, 15
        n = 999
        x = np.linspace(0, 1, n)
        y = f(x, (a, b, c)) + rng.normal(0, 2e9, n)
        p, q, r = self.estimate_initial(x, y, plot=plot)
        self.assertAlmostEqual(p, a, delta=1e10)
        self.assertAlmostEqual(q, b, delta=1e6)
        self.assertAlmostEqual(r, c, delta=2)
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

    def test_estimate_initial_opposing(self):
        f = expfit.exp
        plot = False

        p = 8, 3, -7, -1, -5
        x = np.linspace(0.5, 1.5, 200)
        q = self.estimate_initial_opposing(x, f(x, p), plot=plot)
        self.assertAlmostEqual(q[0], 8, delta=0.001)
        self.assertAlmostEqual(q[1], 3, delta=2)
        self.assertAlmostEqual(q[2], -7, delta=4)
        self.assertAlmostEqual(q[3], -1, delta=2)
        self.assertAlmostEqual(q[4], -5, delta=3)

        p = -3, -200, -10, 10, -6
        x = np.linspace(0.5, 1.5, 200)
        q = self.estimate_initial_opposing(x, f(x, p), plot=plot)
        self.assertAlmostEqual(q[0], -3, delta=0.005)
        self.assertAlmostEqual(q[1], -200, delta=120)
        self.assertAlmostEqual(q[2], -10, delta=2)
        self.assertAlmostEqual(q[3], 10, delta=10)
        self.assertAlmostEqual(q[4], -6, delta=3)

        # Test vetting can be disabled
        x = np.linspace(0, 1, 10)
        expfit.estimate_initial_opposing(x, f(x, p))
        self.assertRaisesRegex(
            ValueError, 'Both arrays in series must have same length',
            expfit.estimate_initial_opposing, x, f(x[1:], p))
        self.assertRaisesRegex(
            ValueError, 'operands could not be broadcast',
            expfit.estimate_initial_opposing, x, f(x[1:], p), vet=False)

        # Test size check
        x = np.linspace(0, 1, 5)
        self.assertRaisesRegex(
            ValueError, 'At least 10 points',
            expfit.estimate_initial_opposing, x, f(x, p))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
