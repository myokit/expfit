#!/usr/bin/env python3
#
# Tests the linear fitting methods
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import unittest

import numpy as np

import expfit


class TestLinear(unittest.TestCase):
    """
    Tests linear fiting methods.
    """

    def test_lsq(self):
        # Test linear least squares

        x = np.array([-5, -2, 0, 0.1, 4, 8, 13])
        y = 4 + 13 * x
        a, b = expfit.least_squares(x, y)
        self.assertEqual(a, 4)
        self.assertEqual(b, 13)

        self.assertRaisesRegex(
            ValueError, 'At least 2 points', expfit.least_squares, [1], [2])

        # Test vetting occurs but can be switched off
        x, y = [3, 2], [1, 1]
        self.assertRaisesRegex(
            ValueError, 'strictly increasing', expfit.least_squares, x, y)
        self.assertRaises(
            TypeError, expfit.least_squares, x, y, vet=False)

    def test_find_linear_segment(self):

        a, b, c = 2, 3, 0.4
        x = np.linspace(0, 1, 100)
        y = a + b * np.exp(c * x)
        min_length = 5
        xx, yy, p, q = expfit.find_linear_segment(x, y, min_length)
        self.assertAlmostEqual(p, a + b, 3)
        self.assertAlmostEqual(q, b * c, 1)
        self.assertSequenceEqual(list(xx), list(x[:len(xx)]))
        self.assertSequenceEqual(list(yy), list(y[:len(yy)]))
        self.assertGreaterEqual(len(xx), min_length)
        self.assertEqual(len(xx), len(yy))
        xx, yy, p, q = expfit.find_linear_segment(x, y, 5, left=False)
        self.assertAlmostEqual(p + q, a + b * np.exp(c), 3)
        self.assertAlmostEqual(q, b * c * np.exp(c), 1)
        self.assertSequenceEqual(list(xx), list(x[-len(xx):]))
        self.assertSequenceEqual(list(yy), list(y[-len(yy):]))

        self.assertRaisesRegex(
            ValueError, 'At least 2 points',
            expfit.find_linear_segment, x[:1], y[:1], 1)

        # Vets, but can be disabled
        self.assertRaisesRegex(
            ValueError, 'must have same length, got 100 and 99',
            expfit.find_linear_segment, x, y[:-1], 5)
        self.assertRaisesRegex(
            ValueError, 'operands could not be broadcast',
            expfit.find_linear_segment, x, y[:-1], 5, vet=False)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
