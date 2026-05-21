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


class TestError(unittest.TestCase):
    """
    Tests transformations.
    """
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


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
