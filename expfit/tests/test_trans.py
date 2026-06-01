#!/usr/bin/env python3
#
# Tests the transformation classes
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import unittest

import numpy as np

import expfit


class TestTrans(unittest.TestCase):
    """
    Tests transformations.
    """
    def test_zoom_transform(self):
        x = np.linspace(0, 1, 111)
        y = expfit.exp(x, (8, 2, 7))

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
        y = expfit.exp(x, (1, 1, 1))
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

    def test_unit_transform(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 9, 1, 8, 7])

        tr = expfit.UnitSquareTransform(x, y)
        a, b, c = 1, 2, 3
        p, q, r = tr.transform(a, b, c)
        self.assertEqual(p, -0.2)
        self.assertAlmostEqual(q, 8.034214769275)
        self.assertEqual(r, 12)
        u, v, w = tr.detransform(p, q, r)
        self.assertEqual(u, a)
        self.assertEqual(v, b)
        self.assertEqual(w, c)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
