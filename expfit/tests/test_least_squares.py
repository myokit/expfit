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


class TestLeastSquares(unittest.TestCase):
    """ Tests linear least squares. """

    def test_ls(self):
        # Test linear least squares

        x = np.array([-5, -2, 0, 0.1, 3, 8, 13])
        y = 4 + 13 * x
        ls = expfit.LeastSquaresFit(x, y)
        self.assertEqual(ls.offset, 4)
        self.assertEqual(ls.slope, 13)
        self.assertEqual(ls.mu_x, np.mean(x))
        self.assertEqual(ls.mu_y, np.mean(y))

        self.assertRaisesRegex(
            ValueError, 'At least 2 points', expfit.LeastSquaresFit, [1], [2])

        # Test vetting occurs but can be switched off
        x, y = [3, 2], [1, 1]
        self.assertRaisesRegex(
            ValueError, 'strictly increasing', expfit.LeastSquaresFit, x, y)
        self.assertRaises(
            TypeError, expfit.LeastSquaresFit, x, y, vet=False)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
