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


class TestFitd2Log(unittest.TestCase):
    """
    Tests fitting of double exponentials with ``fitd2log``.
    """
    @classmethod
    def setUpClass(cls):
        # Create in each test and seed!
        cls.r = None

    def d2_on_double(self, a, b, c, d, e, duration=2, n=200, fnoise=0.01, t0=0,
                     deltas=[], ratio=1, plot=False):
        """
        Tests a double exponential fit to a double exponential signal.

        Criteria: ``digits`` is the ``assertAlmostEqual`` precision with which
        fitted parameters match.

        Criteria: ``deltas`` are the ``assertAlmostEqual`` ``delta`` of the
        parameters, ``ratio`` is the max rmse fit/true ratio, and ``rmse`` is
        the max rmse.
        """
        t = np.linspace(t0, t0 + duration, n)
        v = expfit.exp(t, (a, b, c, d, e))
        v += self.r.normal(0, max(fnoise * abs(v[0] - v[-1]), 1e-9), size=n)

        plot_params = (a, b, c, d, e) if plot else False
        af, bf, cf, df, ef = expfit.fitd2log(t, v, plot=plot_params)
        rt = expfit.rmse(t, v, (a, b, c, d, e))
        rf = expfit.rmse(t, v, (af, bf, cf, df, ef))

        if plot:  # pragma: no cover
            print(f'RMSE true: {rt}')
            print(f'RMSE fit:  {rf}')
            print(f'ratio: {rf / rt}')
            import matplotlib.pyplot as plt
            plt.show()

        with self.subTest(a=a, b=b, c=c, d=d, e=e, duration=duration, n=n,
                          fnoise=fnoise, t0=t0):
            if len(deltas) == 5:
                self.assertAlmostEqual(af, a, delta=deltas[0])
                self.assertAlmostEqual(bf, b, delta=deltas[1])
                self.assertAlmostEqual(cf, c, delta=deltas[2])
                self.assertAlmostEqual(df, d, delta=deltas[3])
                self.assertAlmostEqual(ef, e, delta=deltas[4])
            if ratio is not None:
                self.assertLess(rf / rt, ratio)
            if len(deltas) != 3 and ratio is None:
                raise Exception('No test criteria set')  # pragma: no cover

    def test_fitd2(self):
        # Test double-on-double exponential decaying, equal sign multiplier
        dod = self.d2_on_double
        self.r = np.random.default_rng(20)
        plot = False

        dod(0, -10, -2, -4, -8, deltas=(.05, 1, .1, 1, 2), plot=plot)
        dod(-1e5, 5, -2, 3, -10, deltas=(.05, .5, .2, .5, .5), plot=plot)
        dod(5, 1, -1, 5, -10, deltas=(.1, .1, .5, .2, .5), plot=plot)
        dod(20, 6, -2, 40, -6, t0=0.5, deltas=(.05, 1, .2, 2, .01), plot=plot)
        dod(-87, 30, -3, 40, -20, deltas=(.6, 3, .2, 3, 2), plot=plot)
        dod(123, -8, -1, -5, -99, deltas=(.2, .1, .05, .2, 15), plot=plot)
        dod(400, 5, -1, 3, -4, deltas=(1, .5, .5, 1, 1), plot=plot)
        dod(500, 1, -6, 3, -10, deltas=(.001, 1, 1, 1, 1), plot=plot, n=999)

    def test_fitd2_hard(self):
        # Test cases where it doesn't seem identifiable
        dod = self.d2_on_double
        plot = False

        # Note: These are hard in part because they are sparse. Increasing the
        # number of points we get far better results

        # Noise has strong influence on this one
        # Note that both tests pass the "ratio" criterium: the obtained
        # solution has a lower RMSE than the true solution
        self.r = np.random.default_rng(3)
        dod(17, 10, -6, 5, -12, deltas=(.05, 10, 2, 10, 30), plot=plot)
        dod(17, 10, -6, 5, -12, deltas=(.01, 2, .5, 2, 1), plot=plot, n=999)

        #self.r = np.random.default_rng(2)
        dod(18, 10, -6, 5, -12, deltas=(.05, 5, 1, 5, 3), plot=plot)
        dod(18, 10, -6, 5, -12, deltas=(.01, 1, .2, 1, 1), plot=plot, n=1000)

        #self.r = np.random.default_rng(6)
        dod(100, 10, -2, 4, -5, deltas=(.3, 4, .6, 4, 4), plot=plot)
        dod(100, 10, -2, 4, -5, deltas=(.01, .2, .05, .2, .5), plot=plot,
            n=1000)

        # Unidentifiable? Fits this with 1 time constant
        self.r = np.random.default_rng(3)
        dod(200, -4, -4, -4, -5, deltas=(.05, 5, 2, 5, .5), plot=plot)
        dod(200, -4, -4, -4, -5, deltas=(.1, 4, 4, 5, .5), plot=plot, n=1000)
        self.r = np.random.default_rng(9)
        dod(300, -4, -4, -4, -5, deltas=(.01, 5, 1, 5, 10), plot=plot)
        dod(300, -4, -4, -4, -5, deltas=(.001, 5, 1, 4, 1), plot=plot, n=5000)

        dod(-1e5, 1, -1, 2, -2, deltas=(.5, 1, .5, 1, .5), plot=plot)
        dod(-1e5, 1, -1, 2, -2, deltas=(.01, .2, .1, .2, .1), plot=plot,
            n=8000)

    def test_fitd2_noisy(self):
        # Test on (Gaussian) noisy signals: rapidly becomes impossible
        dod = self.d2_on_double
        self.r = np.random.default_rng(2)
        plot = False

        dod(20, 6, -2, 4, -10, deltas=(.1, .1, .01, .5, .5), plot=plot,
            fnoise=0.05)
        dod(21, 6, -2, 4, -20, deltas=(.5, 1, 1, 1, 20), plot=plot,
            fnoise=0.1)
        dod(-87, 30, -3, 40, -20, deltas=(.5, .5, .1, 2, 2), plot=plot,
            fnoise=0.05)
        dod(123, -8, -1, -5, -99, deltas=(.5, .5, .1, 1, 20), plot=plot,
            fnoise=0.05)

    def test_fitd2_edge_cases(self):

        # Case where scaling to unit square would give a  divide-by-zero
        x = np.linspace(0, 1, 10)
        y = np.zeros(x.shape)
        a, b, c, d, e = expfit.fitd2log(x, y)
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

    #def test_taud2(self):
    #    c1, c2, ci1, ci2 = fitd2_tau


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
