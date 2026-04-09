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

        pass

        # TODO


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
