#!/usr/bin/env python3
#
# Basic tests for the optimiser, including error (the real tests are the fits)
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import unittest

import numpy as np

import expfit


class Poly2():
    """ Quadratic to optimise in x """
    def __init__(self, a, b, c):
        self._abc = a, b, c

    def __call__(self, x):
        x = x[0] if isinstance(x, np.ndarray) else float(x)
        a, b, c = self._abc
        return (a * x**2 + b * x + c, [2 * a * x + b], [[2 * a]])


class Poly4():
    """ Fourth order polynomial to optimise in x """
    def __init__(self, a, b, c, d):
        self._abcd = a, b, c, d

    def __call__(self, x):
        x = x[0] if isinstance(x, np.ndarray) else float(x)
        a, b, c, d = self._abcd
        return (a * x**4 + b * x**2 + c * x + d,
                [4 * a * x**3 + 2 * b * x + c],
                [[12 * a * x**2 + 2 * b]])


class TestOpt(unittest.TestCase):
    """ Tests optimisers """

    def test_least_squares(self):
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

        # Test string
        self.assertEqual(str(ls), 'mu (2.44, 35.8), 4.0 + 13.0 x')

    def test_lm(self):

        # Find minimum of quadratic
        f = Poly2(2, 3, 4)
        self.assertEqual(f(3), (31, [15], [[4]]))
        r = expfit.lm(f, [10])
        self.assertTrue(r.success)
        self.assertEqual(r.x.shape, (1, ))
        self.assertAlmostEqual(r.x[0], -0.75)
        self.assertAlmostEqual(r.error, 23 / 8)

        # Test output
        x = str(r).splitlines()
        self.assertEqual(len(x), 12)
        self.assertEqual(x[0], '     message: Optimisation successful')
        self.assertEqual(x[1], '     success: True')
        self.assertEqual(x[2], '  root error: 1.695582495781317')
        self.assertEqual(x[3], '       error: 2.875')
        self.assertTrue(x[4].startswith('    jacobian: [1'))
        self.assertEqual(x[5], '     hessian: [[4]]')
        self.assertEqual(x[6], '           x: [-0.75]')
        self.assertTrue(x[7].startswith('        gtol: 1'))
        self.assertEqual(x[8], f'  iterations: {r.iterations}')
        self.assertEqual(x[9], f' evaluations: {r.evaluations}')
        self.assertEqual(x[10], f'    accepted: {r.accepted}')
        self.assertTrue(x[11].startswith('        time: 0.0'))

        # Test 2d output
        r.jac = [1, 2]
        r.hes = [[3, 4], [5, 6]]
        x = str(r).splitlines()
        self.assertEqual(len(x), 13)
        self.assertEqual(x[4], '    jacobian: [1 2]')
        self.assertEqual(x[5], '     hessian: [[3 4]')
        self.assertEqual(x[6], '               [5 6]]')

        # Find minimum of 4th order
        f = Poly4(3, 2, 5, 7)
        self.assertEqual(f(2), (73, [109], [[148]]))
        r = expfit.lm(f, [10])
        self.assertTrue(r.success)
        self.assertEqual(r.x.shape, (1, ))
        self.assertAlmostEqual(r.x[0], -0.600471416)
        self.assertGreater(f(r.x[0] + 1e-8)[0], r.error)
        self.assertGreater(f(r.x[0] - 1e-8)[0], r.error)

        # Failing constraint
        r = expfit.lm(f, [10], constraint=lambda p: False)
        self.assertFalse(r.success)
        self.assertEqual(r.message, 'Initial position fails constraint')

    def test_lm_bad_error(self):
        # Test optimiser recognises wrong size mse, jac & hes

        e = lambda p: ([1, 2], [1, 2], [1, 2])
        self.assertRaisesRegex(
            ValueError, 'MSE must be a scalar',
            expfit.lm, e, [1, 2])

        e = lambda p: (1, [1], [1, 2])
        self.assertRaisesRegex(
            ValueError, 'Jacobian must match shape of initial point',
            expfit.lm, e, [1, 2])

        e = lambda p: (1, [1, 2], [[1, 2, 3], [3, 4, 5], [6, 7, 8]])
        self.assertRaisesRegex(
            ValueError, 'Hessian must match shape of initial point',
            expfit.lm, e, [1, 2])


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
