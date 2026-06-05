#!/usr/bin/env python3
#
# Tests for double decaying exponential fits
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import unittest

import numpy as np

import expfit


class TestD11(unittest.TestCase):
    """
    Tests fitting of d11 (one down one up) exponentials.
    """
    @classmethod
    def setUpClass(cls):
        # Create in each test and seed!
        cls.r = None

    def d11_on_d11(self, a, b, c, d, e, s, t0=0, duration=2, n=200,
                   deltas=[], ratio=1, rmse=None, plot=False):
        """
        Tests a d11 fit on a d11 signal.

        Criteria: ``digits`` is the ``assertAlmostEqual`` precision with which
        fitted parameters match.

        Criteria: ``deltas`` are the ``assertAlmostEqual`` ``delta`` of the
        parameters, ``ratio`` is the max rmse fit/true ratio, and ``rmse`` is
        the max rmse.
        """
        t = np.linspace(t0, t0 + duration, n)
        v = expfit.exp(t, (a, b, c, d, e))
        v += self.r.normal(0, s, size=n)

        plot_params = (a, b, c, d, e) if plot else False
        p = expfit.fitd11(t, v, plot=plot_params)
        #rt = expfit.rmse(t, v, (a, b, c, d, e))
        #rf = expfit.rmse(t, v, (af, bf, cf, df, ef))

        if plot:  # pragma: no cover
            #print(f'RMSE true: {rt}')
            #print(f'RMSE fit:  {rf}')
            #print(f'ratio: {rf / rt}')
            import matplotlib.pyplot as plt
            plt.show()

        with self.subTest(a=a, b=b, c=c, d=d, e=e, s=s, t0=t0,
                          duration=duration, n=n):
            #if len(deltas) == 5:
            #    self.assertAlmostEqual(af, a, delta=deltas[0])
            #    self.assertAlmostEqual(bf, b, delta=deltas[1])
            #    self.assertAlmostEqual(cf, c, delta=deltas[2])
            #    self.assertAlmostEqual(df, d, delta=deltas[3])
            #    self.assertAlmostEqual(ef, e, delta=deltas[4])
            #if ratio is not None:
            #    self.assertLess(rf / rt, ratio)
            #if rmse is not None:
            #    self.assertLess(rf, rmse)
            if len(deltas) != 3 and ratio is None and rmse is None:
                raise Exception('No test criteria set')  # pragma: no cover

    def test_fitd11(self):
        # Test d11 on same
        d = self.d11_on_d11
        self.r = np.random.default_rng(101)
        plot = False

        d(10, 20, -15, -20, -2, s=0.2, deltas=(1e-9, 1e-9, 1e-9, 1e-9, 1e-9), plot=False)

        import matplotlib.pyplot as plt
        plt.show()

    '''
    def test_fitd2_edge_cases(self):

        # Case where scaling to unit square would give a  divide-by-zero
        x = np.linspace(0, 1, 10)
        y = np.zeros(x.shape)
        a, b, c, d, e = expfit.fitd2(x, y)
        self.assertEqual(a, 0)
        self.assertEqual(b, 0)
        self.assertEqual(c, 0)
        self.assertEqual(d, 0)
        self.assertEqual(e, 0)

        # Non-decreasing
        x = np.linspace(0, 1, 77)
        y = expfit.exp(x, (1, 2, 3, 4, 5))
        self.assertRaisesRegex(
            RuntimeError, 'not decaying', expfit.fitd2, x, y)
    '''


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
