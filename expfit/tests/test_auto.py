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


    '''

    Ideally a lot of these should fail auto

    def test_fit1_straight(self):
        # Test straight lines are detected
        sos = self.single_on_straight
        plot = False
        self.r = np.random.default_rng(1)

        # Flat line with noise: fits _something_
        s = 1
        x = np.linspace(0, 1, 200)
        y = self.r.normal(0, s, x.shape)
        sos(x, y, fails=False, plot=plot)

        # Flat line with dense noise and offset
        x = np.linspace(0, 1, 3000)
        y = 3 * np.ones(x.shape) + self.r.normal(0, 1e-9, x.shape)
        sos(x, y, fails=False, plot=plot)

        # Straight line through origin, with noise
        x = np.linspace(0, 1, 99)
        y = 3 * x + self.r.normal(0, 0.1, x.shape)
        sos(x, y, fails=False, plot=plot)

        # Straight line with offset and noise.
        self.r = np.random.default_rng(1)
        x = np.linspace(0, 1, 99)
        y = 4 + 2 * x + self.r.normal(0, 0.1, x.shape)
        sos(x, y, fails=False, plot=plot)

        # Depending on the noise, this can still look exponential
        self.r = np.random.default_rng(9)
        y = 4 + 2 * x + self.r.normal(0, 0.1, x.shape)
        sos(x, y, fails=False, plot=plot)

        # Flat with dense noise: very sensitive to seed
        x = np.linspace(0, 1, 900)
        self.r = np.random.default_rng(2)
        y = self.r.normal(0, s, x.shape)
        sos(x, y, fails='Too many successive failed steps', plot=plot)
        self.r = np.random.default_rng(3)
        y = self.r.normal(0, s, x.shape)
        sos(x, y, fails=False, plot=plot)
        self.r = np.random.default_rng(4)
        y = self.r.normal(0, s, x.shape)
        sos(x, y, fails='Maximum iterations reached', plot=plot)

        # Mock-up of situation where noise creates an apparent exponential, but
        # only in the first two samples
        x = np.linspace(0, 1, 20)
        y = np.zeros(x.shape)
        y[0] = 1
        y[1] = 0.1
        sos(x, y, plot=plot)
    '''

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
