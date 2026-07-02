#!/usr/bin/env python3
#
# Tests for the error methods and classes
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import unittest

import numpy as np

import expfit
from expfit.tests import FDError, FDErrorMulti, FDErrorTau


class TestError(unittest.TestCase):
    """ Tests the different error classes. """

    def test_exp1(self):
        # Single exponential function in c-form

        x = np.linspace(0, 1, 123)
        a, b, c = 1, 2, 3
        y = a + b * np.exp(c * x)
        np.testing.assert_array_equal(y, expfit.exp1(x, (a, b, c)))

    def test_expd(self):
        # Decaying exponential function in tau form

        x = np.linspace(0, 1, 123)
        a, b, c = 1, 2, 3
        y = a + b * np.exp(-x / c)
        np.testing.assert_array_equal(y, expfit.expd(x, (a, b, c)))

        x = np.linspace(5, 15, 200)
        a, b, c, d, e = 5, 6, 0.2, 8, 0.3
        y = a + b * np.exp(-x / c) + d * np.exp(-x / e)
        np.testing.assert_allclose(
            y, expfit.expd(x, (a, b, c, d, e)), rtol=1e-15)

        self.assertRaisesRegex(
            ValueError, 'number of parameters', expfit.expd, x, (a, b))

    def test_rmse1(self):
        # RMSE on single exponential in c-form

        x = np.linspace(1, 2, 50)
        p1 = 3, 2, 3
        p2 = 4, 7, 2
        y1 = expfit.exp1(x, p1)
        y2 = expfit.exp1(x, p2)
        r = np.sqrt(np.sum((y1 - y2)**2) / len(y1))
        self.assertEqual(r, expfit.rmse1(x, y1, p2))
        self.assertEqual(r, expfit.rmse1(x, y2, p1))

    def test_rmsed(self):
        # RMSE on multi-exponential in tau form

        x = np.linspace(1, 2, 50)
        p1 = 3, 2, 3
        p2 = 4, 7, 2
        y1 = expfit.expd(x, p1)
        y2 = expfit.expd(x, p2)
        r = np.sqrt(np.sum((y1 - y2)**2) / len(y1))
        self.assertEqual(r, expfit.rmsed(x, y1, p2))
        self.assertEqual(r, expfit.rmsed(x, y2, p1))

        x = np.linspace(5, 15, 2000)
        p1 = 4, 5, -2, 3, -1
        p2 = 3, 3, -7, 5, -5
        y1 = expfit.expd(x, p1)
        y2 = expfit.expd(x, p2)
        r = np.sqrt(np.sum((y1 - y2)**2) / len(y1))
        self.assertEqual(r, expfit.rmsed(x, y1, p2))
        self.assertEqual(r, expfit.rmsed(x, y2, p1))

    def test_single_error(self):
        # Test the single exponential error

        x = np.linspace(0, 1, 123)
        y = expfit.exp1(x, (1, 2, 3))
        e = expfit.SingleExponentialError(expfit.TimeSeries(x, y))
        m, j, h = e((1, 2, 3))
        self.assertAlmostEqual(m, 0)
        self.assertEqual(len(j), 3)
        self.assertAlmostEqual(j[0], 0)
        self.assertAlmostEqual(j[1], 0)
        self.assertAlmostEqual(j[2], 0)
        self.assertEqual(h.shape, (3, 3))
        self.assertEqual(h[0, 0], 2)
        self.assertAlmostEqual(h[0, 1], 12.79230969)
        self.assertAlmostEqual(h[0, 2], 18.47784527)
        self.assertAlmostEqual(h[1, 1], 136.36719398)
        self.assertAlmostEqual(h[1, 2], 229.03766648)
        self.assertAlmostEqual(h[2, 2], 398.51809386)
        self.assertEqual(h[1, 0], h[0, 1])
        self.assertEqual(h[2, 0], h[0, 2])
        self.assertEqual(h[2, 1], h[1, 2])

        m, j, h = e((2, 1, 2))
        self.assertAlmostEqual(m, 148.65542724)
        self.assertEqual(len(j), 3)
        self.assertAlmostEqual(j[0], -17.17916119)
        self.assertAlmostEqual(j[1], -85.9765438)
        self.assertAlmostEqual(j[2], -71.7043942)
        self.assertEqual(h.shape, (3, 3))
        self.assertEqual(h[0, 0], 2)
        self.assertAlmostEqual(h[0, 1], 6.40545818)
        self.assertAlmostEqual(h[0, 2], 4.22073492)
        self.assertAlmostEqual(h[1, 1], 27.03559498)
        self.assertAlmostEqual(h[1, 2], -50.82565376)
        self.assertAlmostEqual(h[2, 2], -44.60674073)
        self.assertEqual(h[1, 0], h[0, 1])
        self.assertEqual(h[2, 0], h[0, 2])
        self.assertEqual(h[2, 1], h[1, 2])

        p = (1.1, 2.2, 3.1)
        fd = FDError(x, y)
        m1, j1, h1 = e(p)
        m2, j2, h2 = fd(p)
        self.assertAlmostEqual(m1, m2)
        self.assertTrue(np.all(np.abs(j1 - j2) < 4e-4))
        self.assertTrue(np.all(np.abs(h1 - h2) < 2e-2))

        p = (0.9, 1.9, 2.9)
        m1, j1, h1 = e(p)
        m2, j2, h2 = fd(p)
        self.assertAlmostEqual(m1, m2)
        self.assertTrue(np.all(np.abs(j1 - j2) < 4e-4))
        self.assertTrue(np.all(np.abs(h1 - h2) < 2e-2))

        # Test mse() method
        self.assertEqual(e(p)[0], e.mse(p))

        # Test n() method
        self.assertEqual(e.n(), len(x))

    def test_multi_error(self):
        # Test the multi exponential error

        # Single error comparison: MSE only
        x = np.linspace(0, 1, 123)
        y = expfit.expd(x, [1, 2, 1 / 3])
        xy = expfit.TimeSeries(x, y)
        e = expfit.MultiExponentialError(xy, 1, 0, True)
        es = expfit.SingleExponentialError(xy)
        p0 = np.array([1, 2, -2])
        q0 = np.array([1, np.log(2), np.log(2)])
        self.assertEqual(e(q0)[0], es(p0)[0])

        # Test mse() method
        self.assertEqual(e(p0)[0], e.mse(p0))

        # Test n() method
        self.assertEqual(e.n(), len(x))

        # Multi with zeros
        e1 = expfit.MultiExponentialError(xy, 1, 0, True)
        e2 = expfit.MultiExponentialError(xy, 2, 0, True)
        m1, j1, h1 = e1((0.9, 1.9, 2.9))
        m2, j2, h2 = e2((0.9, 1.9, 2.9, -np.inf, 0))
        self.assertEqual(m1, m2)
        self.assertTrue(np.all(np.abs(j2[:3] - j1)) == 0)
        self.assertTrue(np.all(np.abs(h2[:3, :3] - h1)) == 0)

        # Multi versus finite differences
        p = np.array([1, 1.1, 0.5, 1.2, 1])
        y = expfit.expd(x, p)
        xy = expfit.TimeSeries(x, y)
        e = expfit.MultiExponentialError(xy, 2, 0, True)
        fd = FDErrorMulti(x, y, 2, 0, True)
        p = np.array([1.1, 1.2, 0.4, 1.3, 0.9])
        m1, j1, h1 = e(p)
        m2, j2, h2 = fd(p)
        self.assertEqual(j1.shape, (5, ))
        self.assertEqual(h1.shape, (5, 5))
        self.assertAlmostEqual(m1, m2)
        self.assertTrue(np.all(np.abs(j1 - j2) < 1e-5))
        self.assertTrue(np.all(np.abs(h1 - h2) < 0.006))

        e = expfit.MultiExponentialError(xy, 2, 1, True)
        fd = FDErrorMulti(x, y, 2, 1, True)
        p = [1.01, 2.1, 1.8, 2.1, 0.7, 1.1, 1.1]
        m1, j1, h1 = e(p)
        m2, j2, h2 = fd(p)
        self.assertEqual(j1.shape, (7, ))
        self.assertEqual(h1.shape, (7, 7))
        self.assertAlmostEqual(m1, m2)
        self.assertTrue(np.all(np.abs(j1 - j2) < 5e-5))
        self.assertTrue(np.all(np.abs(h1 - h2) < 0.01))

        self.assertRaisesRegex(
            ValueError, 'Expecting 7 parameters, got 2.',
            e, (1, 2))
        self.assertRaisesRegex(
            ValueError, 'Expecting 7 parameters, got 8.',
            e, (1, 2, 3, 4, 5, 6, 7, 8))
        self.assertRaisesRegex(
            ValueError, 'Expecting 7 parameters, got 3.',
            e.mse, (1, 2, 3))
        self.assertRaisesRegex(
            ValueError, 'with same sign',
            expfit.MultiExponentialError, xy, -1, 1, True)
        self.assertRaisesRegex(
            ValueError, 'with same sign',
            expfit.MultiExponentialError, xy, 0, 1, True)
        self.assertRaisesRegex(
            ValueError, 'with opposite sign',
            expfit.MultiExponentialError, xy, 1, -1, True)

    def test_multi_error_transform(self):
        # Transformations

        x = np.linspace(0, 1, 123)
        y = expfit.expd(x, [1, 2, 0.3])
        xy = expfit.TimeSeries(x, y)
        e = expfit.MultiExponentialError(xy, 1, 1, False)
        t = np.array([2, -3, 0.5, 5, 0.1])
        p = np.array([2, -3, -2, 5, -10])
        q = np.array([2, np.log(3), np.log(2), np.log(5), np.log(10)])

        r = e.transform(p, False)
        self.assertEqual(q.shape, r.shape)
        self.assertAlmostEqual(q[0], r[0], delta=1e-15)
        self.assertAlmostEqual(q[1], r[1], delta=1e-15)
        self.assertAlmostEqual(q[2], r[2], delta=1e-15)
        self.assertAlmostEqual(q[3], r[3], delta=1e-15)
        self.assertAlmostEqual(q[4], r[4], delta=1e-15)

        r = e.transform(t, True)
        self.assertEqual(q.shape, r.shape)
        self.assertAlmostEqual(q[0], r[0], delta=1e-15)
        self.assertAlmostEqual(q[1], r[1], delta=1e-15)
        self.assertAlmostEqual(q[2], r[2], delta=1e-15)
        self.assertAlmostEqual(q[3], r[3], delta=1e-15)
        self.assertAlmostEqual(q[4], r[4], delta=1e-15)

        r = e.detransform(q, False)
        self.assertEqual(q.shape, r.shape)
        self.assertAlmostEqual(p[0], r[0], delta=1e-15)
        self.assertAlmostEqual(p[1], r[1], delta=1e-15)
        self.assertAlmostEqual(p[2], r[2], delta=1e-15)
        self.assertAlmostEqual(p[3], r[3], delta=1e-15)
        self.assertAlmostEqual(p[4], r[4], delta=1e-14)

        r = e.detransform(q, True)
        self.assertEqual(q.shape, r.shape)
        self.assertAlmostEqual(t[0], r[0], delta=1e-15)
        self.assertAlmostEqual(t[1], r[1], delta=1e-15)
        self.assertAlmostEqual(t[2], r[2], delta=1e-15)
        self.assertAlmostEqual(t[3], r[3], delta=1e-15)
        self.assertAlmostEqual(t[4], r[4], delta=1e-15)

        self.assertRaisesRegex(
            ValueError, 'Expecting 5 parameters, got 3',
            e.transform, [1, 2, 3])
        self.assertRaisesRegex(
            ValueError, 'Expecting 5 parameters, got 6',
            e.detransform, [1, 2, 3, 4, 5, 6])

    def test_tau_error(self):
        # Test the tau form error

        # MSE test against multie error
        x = np.linspace(1, 2, 50)
        y = expfit.expd(x, [2, 1, 0.5])
        xy = expfit.TimeSeries(x, y)
        e = expfit.TauFormError(xy)
        em = expfit.MultiExponentialError(xy, 1, 1, True)
        m1, j1, h1 = e([5, 2, 0.5, -1, 0.25])
        m2, j2, h2 = em([5, np.log(2), np.log(2), np.log(1), np.log(4)])
        self.assertEqual(m1, m2)

        # Test mse() method
        self.assertEqual(e([1, 2, 3, 4, 5])[0], e.mse([1, 2, 3, 4, 5]))

        # Test n() method
        self.assertEqual(e.n(), len(x))

        # Test against finite differences
        p = np.array([1, 1.1, 0.5, 1.2, 1])
        y = expfit.expd(x, p)
        xy = expfit.TimeSeries(x, y)
        e = expfit.TauFormError(xy)
        fd = FDErrorTau(x, y)
        p = np.array([1.1, 1.2, 0.4, 1.3, 0.9])
        m1, j1, h1 = e(p)
        m2, j2, h2 = fd(p)
        self.assertEqual(j1.shape, (5, ))
        self.assertEqual(h1.shape, (5, 5))
        self.assertAlmostEqual(m1, m2)
        self.assertTrue(np.all(np.abs(j1 - j2) < 1e-6))
        self.assertTrue(np.all(np.abs(h1 - h2) < 1e-5))

        p = [1.01, 2.1, 1.8, 2.1, 0.7, 1.1, 1.1]
        m1, j1, h1 = e(p)
        m2, j2, h2 = fd(p)
        self.assertEqual(j1.shape, (7, ))
        self.assertEqual(h1.shape, (7, 7))
        self.assertAlmostEqual(m1, m2)
        self.assertTrue(np.all(np.abs(j1 - j2) < 1e-6))
        self.assertTrue(np.all(np.abs(h1 - h2) < 1e-3))

        self.assertRaisesRegex(
            ValueError, r'Invalid number of parameters \(2\).',
            e, (1, 2))
        self.assertRaisesRegex(
            ValueError, r'Invalid number of parameters \(2\).',
            e.mse, (1, 2))

    def test_fixed_parameter(self):
        # Test the wrapper that fixes a single parameter

        x = np.linspace(0, 1, 123)
        y = expfit.expd(x, (1, 2, 3))
        xy = expfit.TimeSeries(x, y)
        e1 = expfit.MultiExponentialError(xy, 1, 0, False)
        e2 = expfit.ErrorWithFixedParameter(e1, (2, 3, 4), 0)
        m1, j1, h1 = e1((2, 4, 5))
        m2, j2, h2 = e2((4, 5))
        self.assertEqual(j2.shape, (2, ))
        self.assertEqual(h2.shape, (2, 2))
        self.assertTrue(np.all(np.abs(j1[1:] - j2) == 0))
        self.assertTrue(np.all(np.abs(h1[1:, 1:] - h2) == 0))

        e2 = expfit.ErrorWithFixedParameter(e1, (2, 3, 4), 1)
        m1, j1, h1 = e1((2, 3, 5))
        m2, j2, h2 = e2((2, 5))
        self.assertEqual(j2.shape, (2, ))
        self.assertEqual(h2.shape, (2, 2))
        self.assertEqual(m1, m2)
        j3, h3 = np.delete(j1, 1), np.delete(np.delete(h1, 1, 0), 1, 1)
        self.assertTrue(np.all(np.abs(j3 - j2) == 0))
        self.assertTrue(np.all(np.abs(h3 - h2) == 0))

        e2 = expfit.ErrorWithFixedParameter(e1, (0, 1, 2), 2)
        m1, j1, h1 = e1((0, 1, 2))
        m2, j2, h2 = e2((0, 1))
        self.assertEqual(j2.shape, (2, ))
        self.assertEqual(h2.shape, (2, 2))
        self.assertTrue(np.all(np.abs(j1[:-1] - j2) == 0))
        self.assertTrue(np.all(np.abs(h1[:-1, :-1] - h2) == 0))

        # Test n() method
        self.assertEqual(e2.n(), e1.n())


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
