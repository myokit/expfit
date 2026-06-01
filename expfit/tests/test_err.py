#!/usr/bin/env python3
#
# Tests the error methods and classes
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import unittest

import numpy as np

import expfit


def mse(x, y, p):
    """ Mean-squared error """
    d = len(p)
    assert (d - 1) % 2 == 0 and d > 1
    m = (d - 1) // 2
    p = np.asarray(p)
    bs = p[1::2].reshape((m, 1))        # (m, 1)
    cs = p[2::2].reshape((m, 1))        # (m, 1)
    return np.sum((p[0] - y + np.sum(bs * np.exp(cs * x), axis=0))**2) / len(x)


def mse_jac_fd(x, y, p, dp):
    """ MSE plus jacobian through finite differences """
    e = mse(x, y, p)
    jac = np.zeros(len(p))
    p = np.array(p, dtype=float)
    for i in range(len(p)):
        q = np.copy(p)
        q[i] += dp
        jac[i] = (mse(x, y, q) - e) / dp
    return e, jac


def mse_jac_hes_fd(x, y, p, dp=1e-6):
    """ MSE plus jacobian and hessian through finite differences """
    d = len(p)
    mse, jac = mse_jac_fd(x, y, p, dp)
    hes = np.zeros((d, d))
    p = np.array(p, dtype=float)
    for i in range(len(p)):
        q = np.copy(p)
        q[i] += dp
        hes[i] = (mse_jac_fd(x, y, q, dp)[1] - jac) / dp
    return mse, jac, hes


class TestError(unittest.TestCase):
    """ Tests the different error classes. """

    def test_exp(self):
        x = np.linspace(0, 1, 123)
        a, b, c = 1, 2, 3
        y = a + b * np.exp(c * x)
        np.testing.assert_array_equal(y, expfit.exp(x, (a, b, c)))

        x = np.linspace(5, 15, 2000)
        a, b, c, d, e = 5, 6, -7, 8, -9
        y = a + b * np.exp(c * x) + d * np.exp(e * x)
        np.testing.assert_array_equal(y, expfit.exp(x, (a, b, c, d, e)))

        x = [1, 2, 3]
        y = expfit.exp(x, [4])
        self.assertEqual(list(y), [4, 4, 4])

    def test_rmse(self):
        x = np.linspace(1, 2, 50)
        p1 = 3, 2, 3
        p2 = 4, 7, 2
        y1 = expfit.exp(x, p1)
        y2 = expfit.exp(x, p2)
        r = np.sqrt(np.sum((y1 - y2)**2) / len(y1))
        self.assertEqual(r, expfit.rmse(x, y1, p2))
        self.assertEqual(r, expfit.rmse(x, y2, p1))

        x = np.linspace(5, 15, 2000)
        p1 = 4, 5, -2, 3, -1
        p2 = 3, 3, -7, 5, -5
        y1 = expfit.exp(x, p1)
        y2 = expfit.exp(x, p2)
        r = np.sqrt(np.sum((y1 - y2)**2) / len(y1))
        self.assertEqual(r, expfit.rmse(x, y1, p2))
        self.assertEqual(r, expfit.rmse(x, y2, p1))

    def test_single_error(self):
        x = np.linspace(0, 1, 123)
        y = expfit.exp(x, (1, 2, 3))
        e = expfit.SingleExponentialError(x, y)
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
        m1, j1, h1 = e(p)
        m2, j2, h2 = mse_jac_hes_fd(x, y, p)
        self.assertAlmostEqual(m1, m2)
        self.assertTrue(np.all(np.abs(j1 - j2) < 4e-4))
        self.assertTrue(np.all(np.abs(h1 - h2) < 2e-2))

        p = (0.9, 1.9, 2.9)
        m1, j1, h1 = e(p)
        m2, j2, h2 = mse_jac_hes_fd(x, y, p)
        self.assertAlmostEqual(m1, m2)
        self.assertTrue(np.all(np.abs(j1 - j2) < 4e-4))
        self.assertTrue(np.all(np.abs(h1 - h2) < 2e-2))

    def test_multi_error(self):

        # Single error comparison
        x = np.linspace(0, 1, 123)
        y = expfit.exp(x, (1, 2, 3))
        e1 = expfit.MultiExponentialError(x, y)
        e2 = expfit.SingleExponentialError(x, y)
        p = (1, 2, 3)
        m1, j1, h1 = e1(p)
        m2, j2, h2 = e2(p)
        self.assertEqual(m1, m2)
        self.assertTrue(np.all(np.abs(j1 - j2) == 0))
        self.assertTrue(np.all(np.abs(h1 - h2) == 0))

        p = (2, 1, 2)
        m1, j1, h1 = e1(p)
        m2, j2, h2 = e2(p)
        self.assertEqual(m1, m2)
        self.assertTrue(np.all(np.abs(j1 - j2) == 0))
        self.assertTrue(np.all(np.abs(h1 - h2) == 0))

        p = (1.1, 2.2, 3.1)
        m1, j1, h1 = e1(p)
        m2, j2, h2 = e2(p)
        self.assertEqual(m1, m2)
        self.assertTrue(np.all(np.abs(j1 - j2) < 1e-14))
        self.assertTrue(np.all(np.abs(h1 - h2) < 1e-14))

        p = (0.9, 1.9, 2.9)
        m1, j1, h1 = e1(p)
        m2, j2, h2 = e2(p)
        self.assertEqual(m1, m2)
        self.assertTrue(np.all(np.abs(j1 - j2) == 0))
        self.assertTrue(np.all(np.abs(h1 - h2) < 5e-14))

        # Multi with zeros
        m1, j1, h1 = e1((0.9, 1.9, 2.9))
        m2, j2, h2 = e1((0.9, 1.9, 2.9, 0, 0))
        self.assertEqual(m1, m2)
        self.assertTrue(np.all(np.abs(j2[:3] - j1)) == 0)
        self.assertTrue(np.all(np.abs(h2[:3, :3] - h1)) == 0)

        # Multi versus finite differences
        p = (1.2, 2.2, 1.9, 3.2, 1.8)
        m1, j1, h1 = e1(p)
        m2, j2, h2 = mse_jac_hes_fd(x, y, p)
        self.assertEqual(j1.shape, (5, ))
        self.assertEqual(h1.shape, (5, 5))
        self.assertAlmostEqual(m1, m2)
        self.assertTrue(np.all(np.abs(j1 - j2) < 1e-4))
        self.assertTrue(np.all(np.abs(h1 - h2) < 6e-3))

        p = (1.01, 2.1, 1.8, 2.1, 0.7, 1.2, 0.8)
        m1, j1, h1 = e1(p)
        m2, j2, h2 = mse_jac_hes_fd(x, y, p)
        self.assertEqual(j1.shape, (7, ))
        self.assertEqual(h1.shape, (7, 7))
        self.assertAlmostEqual(m1, m2)
        self.assertTrue(np.all(np.abs(j1 - j2) < 1e-5))
        self.assertTrue(np.all(np.abs(h1 - h2) < 0.03))

    def test_fixed_parameter(self):
        x = np.linspace(0, 1, 123)
        y = expfit.exp(x, (1, 2, 3))
        e1 = expfit.MultiExponentialError(x, y)
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


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
