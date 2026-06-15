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


class TestDouble(unittest.TestCase):
    """
    Tests fitting of double exponentials.
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
        af, bf, cf, df, ef = expfit.fitd2(t, v, plot=plot_params)
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

        dod(0, -10, 0.5, -4, 0.125, deltas=(.05, 1, .01, 1, .02), plot=plot)
        dod(-1e5, 5, 0.5, 3, 0.1, deltas=(.05, .5, .01, .5, 1e-3), plot=plot)
        dod(5, 1, 1, 5, 0.1, deltas=(.1, .1, .2, .2, 2e-3), plot=plot)
        dod(20, 6, 0.5, 40, .17, t0=0.5, deltas=(.05, 1, .05, 2, 2e-3),
            plot=plot)
        dod(-87, 30, .33, 40, .05, deltas=(.6, 3, .05, 3, 2e-3), plot=plot)
        dod(123, -8, 1, -5, .01, deltas=(.2, .1, .05, .2, 1e-3), plot=plot)
        dod(400, 5, 1, 3, .25, deltas=(1, .5, .2, 1, .05), plot=plot)
        dod(500, 1, .17, 3, .1, deltas=(.001, 1, .02, 1, 5e-3), plot=plot,
            n=999)

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
        dod(17, 10, .17, 5, .08, deltas=(.05, 10, .02, 10, .05), plot=plot)
        dod(17, 10, .17, 5, .08, deltas=(.01, 2, .5, 2, .01), plot=plot, n=999)

        self.r = np.random.default_rng(2)
        dod(18, 10, .17, 5, .08, deltas=(.05, 5, .5, 5, .1), plot=plot)

        # Fast component is too small
        self.r = np.random.default_rng(6)
        #dod(100, 10, .25, 4, .2, deltas=(.3, 4, 1e-9, 4, 1e-9), plot=True) # TODO
        dod(100, 10, .25, 4, .2, deltas=(.01, 5, .02, 4, .2), plot=plot, n=999)

        # Unidentifiable? Fits this with 1 time constant
        self.r = np.random.default_rng(3)
        #dod(200, -4, .25, -4, .2, deltas=(.05, 5, .1, 5, .005), plot=plot)  # TODO
        #dod(200, -4, .25, -4, .2, deltas=(.1, 4, 1e-9, 5, 1e-9), plot=True, n=1000)  # TODO

        # These two are repeated in _slow
        self.r = np.random.default_rng(9)
        # dod(300, -4, .25, -4, .2, deltas=(.01, 5, .02, 5, .2), plot=plot) # TODO
        #dod(-1e5, 1, 1, 2, .5, deltas=(.5, 1, .5, 1, .1), plot=plot)  # TODO

    def test_fitd2_slow(self):
        # Hard fits that run very slowly
        dod = self.d2_on_double
        plot = False

        self.r = np.random.default_rng(9)
        #dod(300, -4, .25, -4, .2, deltas=(.001, 5, .1, 4, .02), plot=plot, n=5000)  # TODO
        #dod(-1e5, 1, 1, 2, .5, deltas=(.05, .2, .2, .2, .02), plot=plot,  # TODO
        #    n=8000)
        return

    def test_fitd2_noisy(self):
        # Test on (Gaussian) noisy signals: rapidly becomes impossible
        dod = self.d2_on_double
        self.r = np.random.default_rng(2)
        plot = False

        dod(20, 6, .5, 4, .1, deltas=(.1, .1, .005, .5, .005), plot=plot,
            fnoise=0.05)
        dod(21, 6, .5, 4, .05, deltas=(.5, 1, .2, 1, .05), plot=plot,
            fnoise=0.1)
        dod(-87, 30, .34, 40, .05, deltas=(.5, .5, .005, 2, .005), plot=plot,
            fnoise=0.05)
        dod(123, -8, 1, -5, .01, deltas=(.5, .5, .1, 1, .005), plot=plot,
            fnoise=0.05)

    def test_fitd2_edge_cases(self):

        # Case where scaling to unit square would give a  divide-by-zero
        x = np.linspace(0, 1, 10)
        y = np.zeros(x.shape)
        self.assertRaises(expfit.NotExponentialError, expfit.fitd2, x, y)

        # Non-decreasing
        x = np.linspace(0, 1, 77)
        y = expfit.exp(x, (1, 2, -3))
        self.assertRaises(expfit.NotDecayingError, expfit.fitd2, x, y)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
