#!/usr/bin/env python3
#
# Real world tests for intial estimates, based on earlier failures.
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import unittest

import expfit


class TestEstRw(unittest.TestCase):
    """ Real world tests for initial estimates. """

    def test_rw1(self):
        plot = False

        # With this data, the first c >> 0, causing an overflow that should be
        # caught. The eventual data is better fit by a straight line.
        x = [0.9632, 0.96655, 0.96989, 0.97324, 0.97658, 0.97993,
             0.9832, 0.98662, 0.98996, 0.99331, 0.99665, 1.0]
        y = [0.0,    0.00829, 0.00284, 0.01651, 0.00950, 0.01380,
             0.0083, 0.00883, 0.00040, 0.01087, 0.01880, 0.00245]
        self.assertRaisesRegex(expfit.NotExponentialError, 'Straight line',
                               expfit.est1, x, y, plot=plot)

        # With this data, the first c << 0, causing a divide by zero that
        # should be caught. The solution found is now slightly better fit by an
        # exponential than a straight line.
        for i in range(6):
            y[i] -= 0.001
        p0 = expfit.est1(x, y, plot=plot)

        # This one doesn't cause e1 == e2, but does cause them both to be too
        # large for a division
        x = [0.98996656, 0.99331104, 0.99665552, 1.]
        y = [0., 0.00501364, 0.00876011, 0.00032179]
        self.assertRaisesRegex(expfit.NotExponentialError, 'Straight line',
                               expfit.est1, x, y, plot=plot)

        if plot:
            import matplotlib.pyplot as plt
            plt.show()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
