#!/usr/bin/env python3
#
# Tests for expfit.auto
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import unittest

import numpy as np

import expfit


class TestAuto(unittest.TestCase):
    """
    Tests automatic determinination of number of exponentials.
    """
    @classmethod
    def setUpClass(cls):
        # Create in each test and seed!
        cls.r = None

    def test_auto(self):
        # Test double-on-double exponential decaying, equal sign multiplier
        self.r = np.random.default_rng(1)
        plot = True


        p0 = 5, 5, 5, 5, 1, 5, .1
        # p0 = 5, 10, 2, 5, .5, 5, .1
        #p0 = 5, 10, 5, 5, 1, 5, .3
        # p0 = 5, 10, 5, 5, 1, 5, .3, 10, .05

        t = np.linspace(0, 5, 300, endpoint=False)


        if True:
            v = expfit.exp(t, p0) + self.r.normal(0, 0.1, size=t.shape)
            expfit.auto(t, v, plot=p0, opt_plot=True)
        elif plot:
            from expfit._plot import exp_plot
            exp_plot(t, p0)


        if plot:  # pragma: no cover
            import matplotlib.pyplot as plt
            plt.show()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
