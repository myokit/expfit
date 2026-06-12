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


class TestD11Log(unittest.TestCase):
    """
    Tests fitting of d11 (one down one up) exponentials. LOG LOG LOG
    """
    @classmethod
    def setUpClass(cls):
        # Create in each test and seed!
        cls.r = None

    def d11_on_d11(self, a, b, c, d, e, s, t0=0, duration=2, n=200,
                   deltas=[], ratio=1, plot=False):
        """
        Tests a d11 fit on a d11 signal.

        Criteria: ``digits`` is the ``assertAlmostEqual`` precision with which
        fitted parameters match.

        Criteria: ``deltas`` are the ``assertAlmostEqual`` ``delta`` of the
        parameters, ``ratio`` is the max rmse fit/true ratio.
        """
        t = np.linspace(t0, t0 + duration, n)
        v = expfit.exp(t, (a, b, c, d, e))
        v += self.r.normal(0, s, size=n)

        plot_params = (a, b, c, d, e) if plot else False
        p = expfit.fitd11log(t, v, plot=plot_params)
        rt = expfit.rmse(t, v, (a, b, c, d, e))
        rf = expfit.rmse(t, v, p)

        if plot:  # pragma: no cover
            print(f'RMSE true: {rt}')
            print(f'RMSE fit:  {rf}')
            print(f'ratio: {rf / rt}')
            import matplotlib.pyplot as plt
            plt.show()

        with self.subTest(a=a, b=b, c=c, d=d, e=e, s=s, t0=t0,
                          duration=duration, n=n):
            if len(deltas) == 5:
                self.assertAlmostEqual(p[0], a, delta=deltas[0])
                self.assertAlmostEqual(p[1], b, delta=deltas[1])
                self.assertAlmostEqual(p[2], c, delta=deltas[2])
                self.assertAlmostEqual(p[3], d, delta=deltas[3])
                self.assertAlmostEqual(p[4], e, delta=deltas[4])
            if ratio is not None:
                self.assertLess(rf / rt, ratio)
            if len(deltas) != 3 and ratio is None:   # pragma: no cover
                raise Exception('No test criteria set')

    def test_fitd11(self):
        # Test d11 on same
        d = self.d11_on_d11
        self.r = np.random.default_rng(101)
        plot = True

        d(10, 20, -15, -20, -2, s=0.2, deltas=(.1, .2, .5, .02, .01),
          plot=plot)
        d(10, 80, -15, -20, -2, s=0.2, deltas=(.1, .1, .001, .1, .1),
          plot=plot)
        d(10, 40, -15, -20, -2, s=0.2, t0=0.1, deltas=(.1, 2, .5, .2, .1),
          plot=plot)
        d(7, -100, -10, 15, -5, s=0.2, t0=0.1, deltas=(.1, 10, 1, 6, 1),
          plot=plot)
        d(5, -10, -15, 10, -2, s=0.2, t0=0.1, deltas=(.1, 1, 1, .5, .1),
          plot=plot)

    def d11_on_d12(self, p, s, t0=0, duration=2, n=100, ratio=1, plot=False):
        """ Tests a d11 fit on a d12 signal. """
        t = np.linspace(t0, t0 + duration, n)
        v = expfit.exp(t, p)
        v += self.r.normal(0, s, size=n)

        q = expfit.fitd11log(t, v, plot=p if plot else False)
        rt = expfit.rmse(t, v, p)
        rf = expfit.rmse(t, v, q)

        if plot:  # pragma: no cover
            print(f'RMSE true: {rt}')
            print(f'RMSE fit:  {rf}')
            print(f'ratio: {rf / rt}')
            import matplotlib.pyplot as plt
            plt.show()

        with self.subTest(p=p, s=s, t0=t0, duration=duration, n=n):
            self.assertLess(rf / rt, ratio)

    def test_fitd11_on_d12(self):
        # Test on a double second exponential

        d = self.d11_on_d12
        self.r = np.random.default_rng(101)
        plot = False

        d((1, 6, -5, -4, -2, -2, -4), s=0.01, plot=plot)
        d((1, 6, -5, -4, -2, -2, -4), s=0.1, n=500, plot=plot)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
