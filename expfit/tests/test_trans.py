#!/usr/bin/env python3
#
# Tests for the transformation classes used in single exponential fits
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
    def test_unit_transform(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 9, 1, 8, 7])

        tr = expfit.UnitSquareTransform(x, y)
        a, b, c = 1, 2, 3
        p, q, r = tr.transform(a, b, c)
        self.assertEqual(p, 0)
        self.assertAlmostEqual(q, 5.02138423)
        self.assertEqual(r, 12)
        u, v, w = tr.detransform(p, q, r)
        self.assertEqual(u, a)
        self.assertEqual(v, b)
        self.assertEqual(w, c)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
