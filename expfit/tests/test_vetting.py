#!/usr/bin/env python3
#
# Tests the sequence vetting and conversion to nd array method
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import unittest

import numpy as np

import expfit


class TestVetting(unittest.TestCase):
    """
    Tests array / sequence vetting methods.
    """
    def test_vet_series(self):

        # Increasing arrays, no changes
        x = np.array([-50, -21, 0, 0.1, 4, 8, 13])
        y = np.array([1, 99, -3, -3, -3, 8, 12])
        t, v = expfit.vet_series(x, y)
        self.assertIs(t, x)
        self.assertIs(v, y)

        # Time not strictly increasing
        self.assertRaisesRegex(
            ValueError, 'strictly increasing',
            expfit.vet_series, y, x)

        # Size-1 counts as increasing
        t, v = expfit.vet_series([1], [2])

        # Not the same size
        self.assertRaisesRegex(
            ValueError, 'same length, got 7 and 5',
            expfit.vet_series, x, y[2:])
        self.assertRaisesRegex(
            ValueError, 'same length, got 4 and 6',
            expfit.vet_series, x[3:], y[:-1])

        # Other sequence types are converted
        x = [1, 2, 3]
        y = [8, -1, 4]
        t, v = expfit.vet_series(x, y)
        self.assertIsNot(t, x)
        self.assertIsNot(v, y)
        self.assertEqual(list(t), x)
        self.assertEqual(list(v), y)
        self.assertIsInstance(t, np.ndarray)
        self.assertIsInstance(v, np.ndarray)
        self.assertEqual(t.shape, (3, ))
        self.assertEqual(v.shape, (3, ))

        # But only for numbers
        x = [1, None, False]
        self.assertRaises(TypeError, expfit.vet_series, x, y)

        # Passing in a string? Numpy says its scalar will try to convert to
        # float, and if succesful return length-1 array
        x = 'abc'
        self.assertRaisesRegex(
            ValueError, 'could not convert string to float: \'abc\'',
            expfit.vet_series, x, y)
        x = '1 2 3'
        self.assertRaisesRegex(
            ValueError, 'could not convert string to float: \'1 2 3\'',
            expfit.vet_series, x, y)
        x = '123'
        self.assertRaisesRegex(
            ValueError, 'same length, got 1 and 3',
            expfit.vet_series, x, y)
        expfit.vet_series('1', '2')

        # Multidimensional is brought down to 1, if all other dimensions are 1
        x = np.array([1, 2, 3])
        y = 3 * x
        t, v = expfit.vet_series(
            x.reshape((1, 1, 1, 1, 3, 1, 1)), y.reshape((3, 1, 1)))
        self.assertEqual(t.shape, (3, ))
        self.assertEqual(v.shape, (3, ))
        self.assertRaisesRegex(
            ValueError, 'Unable to convert to 1d vector',
            expfit.vet_series, [[1, 2, 3], [4, 5, 6]], [3, 2, 1])
        self.assertRaisesRegex(
            ValueError, 'Unable to convert to 1d vector',
            expfit.vet_series, [[1, 2], [6, 3]], [3, 2, 1, 2])

        # Scalars turned into length-1
        t, v = expfit.vet_series(1, 2)
        self.assertIsInstance(t, np.ndarray)
        self.assertIsInstance(v, np.ndarray)
        self.assertEqual(list(t), [1])
        self.assertEqual(list(v), [2])
        self.assertEqual(t.shape, (1, ))
        self.assertEqual(v.shape, (1, ))

        # Empty array stays empty, but gets a dimension
        x, y = np.array(0), np.array(0)
        self.assertEqual(x.shape, ())
        self.assertEqual(x.ndim, 0)
        t, v = expfit.vet_series(x, y)
        self.assertEqual(t.shape, (0, ))
        self.assertEqual(t.ndim, 1)
        self.assertEqual(v.shape, (0, ))
        self.assertEqual(v.ndim, 1)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
