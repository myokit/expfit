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

    #@unittest.skip
    def test_auto(self):
        # Test double-on-double exponential decaying, equal sign multiplier
        self.r = np.random.default_rng(1)
        plot = True


        p0 = 5, 5, 5
        p0 = 5, 5, 5, 5, 1
        p0 = 5, 5, 5, 5, 1, 5, .1
        p0 = 5, 10, 2, 5, .5, 5, .1
        p0 = 5, 10, 5, 5, 1, 5, .3
        p0 = 5, 10, 5, 5, 1, 5, .3, 10, .05
        x = np.linspace(0, 5, 300, endpoint=False)

        p0 = 5, -10, 10, -10, 5, 15, 0.5
        x = np.linspace(0, 15, 700, endpoint=False)




        if True:
            y = expfit.expd(x, p0) + self.r.normal(0, 0.1, size=x.shape)
            expfit.auto(x, y, plot=p0, opt_plot=False)
        elif plot:
            from expfit._plot import expd_plot
            expd_plot(x, p0)


        if plot:  # pragma: no cover
            import matplotlib.pyplot as plt
            plt.show()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
