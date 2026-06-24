#!/usr/bin/env python3
#
# Tests for the sequence vetting methods
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import unittest

import numpy as np

import expfit


class TestTimeSeries(unittest.TestCase):
    """ Tests the TimeSeries object and vetting. """

    def test_vetting(self):
        # Test creation of time series and array vetting

        # Increasing arrays, no changes
        x = np.array([-50, -21, 0, 0.1, 4, 8, 13])
        y = np.array([1, 99, -3, -3, -3, 8, 12])
        t, v = expfit.TimeSeries(x, y)
        self.assertIs(t, x)
        self.assertIsNot(v, y)
        self.assertEqual(list(y), list(v))
        self.assertEqual(t.dtype, float)
        self.assertEqual(v.dtype, float)

        # Time not strictly increasing
        self.assertRaisesRegex(
            ValueError, 'strictly increasing',
            expfit.TimeSeries, y, x)

        # Size-1 counts as increasing
        t, v = expfit.TimeSeries([1], [2])

        # Not the same size
        self.assertRaisesRegex(
            ValueError, 'same length, got 7 and 5',
            expfit.TimeSeries, x, y[2:])
        self.assertRaisesRegex(
            ValueError, 'same length, got 4 and 6',
            expfit.TimeSeries, x[3:], y[:-1])

        # Other sequence types are converted
        x = [1, 2, 3]
        y = [8, -1, 4]
        t, v = expfit.TimeSeries(x, y)
        self.assertIsNot(t, x)
        self.assertIsNot(v, y)
        self.assertEqual(list(t), x)
        self.assertEqual(list(v), y)
        self.assertIsInstance(t, np.ndarray)
        self.assertIsInstance(v, np.ndarray)
        self.assertEqual(t.shape, (3, ))
        self.assertEqual(v.shape, (3, ))

        # Must be finite
        x = [1, None, False]
        self.assertRaisesRegex(
            ValueError, 'finite', expfit.TimeSeries, x, y)
        x = [1, np.inf, 2]
        self.assertRaisesRegex(
            ValueError, 'finite', expfit.TimeSeries, x, y)
        x = [1, 2, -np.inf]
        self.assertRaisesRegex(
            ValueError, 'finite', expfit.TimeSeries, x, y)

        # Passing in a string? Numpy says its scalar will try to convert to
        # float, and if succesful return length-1 array
        x = 'abc'
        self.assertRaisesRegex(
            ValueError, 'could not convert string to float: \'abc\'',
            expfit.TimeSeries, x, y)
        x = '1 2 3'
        self.assertRaisesRegex(
            ValueError, 'could not convert string to float: \'1 2 3\'',
            expfit.TimeSeries, x, y)
        x = '123'
        self.assertRaisesRegex(
            ValueError, 'same length, got 1 and 3',
            expfit.TimeSeries, x, y)
        expfit.TimeSeries('1', '2')

        # Multidimensional is brought down to 1, if all other dimensions are 1
        x = np.array([1, 2, 3])
        y = 3 * x
        t, v = expfit.TimeSeries(
            x.reshape((1, 1, 1, 1, 3, 1, 1)), y.reshape((3, 1, 1)))
        self.assertEqual(t.shape, (3, ))
        self.assertEqual(v.shape, (3, ))
        self.assertRaisesRegex(
            ValueError, 'Unable to convert array in time series to 1d vector',
            expfit.TimeSeries, [[1, 2, 3], [4, 5, 6]], [3, 2, 1])
        self.assertRaisesRegex(
            ValueError, 'Unable to convert array in time series to 1d vector',
            expfit.TimeSeries, [[1, 2], [6, 3]], [3, 2, 1, 2])

        # Scalars turned into length-1
        t, v = expfit.TimeSeries(1, 2)
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
        t, v = expfit.TimeSeries(x, y)
        self.assertEqual(t.shape, (0, ))
        self.assertEqual(t.ndim, 1)
        self.assertEqual(v.shape, (0, ))
        self.assertEqual(v.ndim, 1)

    def test_from_tv(self):
        # Test creation with _from_tv

        t, v = np.array([1, 2, 3]), np.array([4, 5, 6])
        t1 = expfit.TimeSeries._from_tv(t, v)
        self.assertIsInstance(t1, expfit.TimeSeries)
        t2 = expfit.TimeSeries._from_tv(t1)
        self.assertIs(t1, t2)
        self.assertRaisesRegex(
            ValueError, 'is not None', expfit.TimeSeries._from_tv, t1, v)
        self.assertRaisesRegex(
            ValueError, 'is None', expfit.TimeSeries._from_tv, t)

    def test_unit_transformed(self):

        # t0=1, v0=1, rt=4, rv=8
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 9, 1, 8, 7])
        tr = expfit.UnitSquareTransformedTimeSeries(x, y)
        a, b, t = 1, 2, 3
        p, q, r = tr.transform(a, b, t)
        self.assertEqual(p, (a - 1) / 8)
        self.assertEqual(q, b / 8 * np.exp(-1 / t))
        self.assertEqual(r, t / 4)
        u, v, w = tr.detransform(p, q, r)
        self.assertEqual(u, a)
        self.assertEqual(v, b)
        self.assertEqual(w, t)
        x, y = tr
        self.assertEqual(list(x), [0, 0.25, 0.5, 0.75, 1])
        self.assertEqual(list(y), [0.125, 1, 0, .875, 0.75])

        # t0=0, v0=-1, rt=1, rv=10
        x = np.array([0, 0.5, 1])
        y = np.array([1, -1, 9])
        tr = expfit.UnitSquareTransformedTimeSeries(x, y)
        a, b, t = 4, 2, -0.5
        p, q, r = tr.transform(a, b, t)
        self.assertEqual(p, (a + 1) / 10)
        self.assertEqual(q, b / 10)
        self.assertEqual(r, t)
        u, v, w = tr.detransform(p, q, r)
        self.assertEqual(u, a)
        self.assertEqual(v, b)
        self.assertEqual(w, t)
        x, y = tr
        self.assertEqual(list(x), [0, 0.5, 1])
        self.assertEqual(list(y), [0.2, 0, 1])


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
