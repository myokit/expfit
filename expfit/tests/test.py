#!/usr/bin/env python3
#
# Tests the WHAT DOES IT DOOOOOOOOOOOOOOOO
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
#import io
#import os
#import re
import unittest

import numpy as np

import expfit

#from myokit.tests import DIR_DATA, WarningCollector

debug = True
if debug:
    import matplotlib.pyplot as plt



def least_squares(x, y):
    """
    Returns a least squares fit ``(a, b)`` where ``y`` is approximated by
    ``a + b * y``.
    """
    # TODO: Input check, vector, same size etc.
    n = len(x)
    mu_x, mu_y = np.mean(x), np.mean(y)
    xx = np.sum(x**2) - n * mu_x**2
    xy = np.sum(x * y) - n * mu_x * mu_y
    b = xy / xx
    return mu_y - b * mu_x, b


def estimate_initial(x, y):
    """
    Estimate initial ``a, b, c`` in ``y = a + b exp(c x)`` using
    derivatives estimated from mean averages at the sides.





    """
    # This but better might be good
    if False:
        # First, check if we're looking at an extremely steep function (so
        # slope of 0 on one side). If so, zoom in on part of trace where
        # the action happens
        i = len(x) // 2
        while i > 4:
            qlo, qhi = np.abs(y[0] - y[i]), np.abs(y[i] - y[-1])
            q = np.log10((qhi / qlo) if qhi > qlo else (qlo / qhi))
            print(f'LR: {q} ({qlo}, {qhi})')
            if q < 2:   # Empirical
                break
            x, y = (x[i:], y[i:]) if qhi > qlo else (x[:-i], y[:-i])
            i = len(x) // 2
        print(f'Final LR: {q}, segment length {len(x)}')

    def inner(x, y):
        n = len(x)

        # Number of samples to use in averaging
        if len(x) < 10:
            raise ValueError('TODO: Complain array too short')
        i = max(3, n // 10)

        # Get points on the left and right, along with slope estimates
        xlo, ylo = np.mean(x[:i]), np.mean(y[:i])
        zlo, slo = least_squares(x[:i], y[:i])
        xhi, yhi = np.mean(x[-i:]), np.mean(y[-i:])
        zhi, shi = least_squares(x[-i:], y[-i:])
        # Note: zlo and zhi are only for plotting

        # Flat line?
        if slo == shi or ylo == yhi:
            print('Warning: initial estimate suggests flat line')
            return np.mean(y), 0, 0

        if False:
            fig = plt.figure(figsize=(8, 5))
            fig.subplots_adjust(0.08, 0.08, 0.99, 0.99)
            ax = fig.add_subplot()
            ax.plot(x, y)
            ax.plot(xlo, ylo, 'ks')
            ax.plot(x[:i], zlo + slo * x[:i], 'k')
            ax.plot(xhi, yhi, 'rs')
            ax.plot(x[-i:], zhi + shi * x[-i:], 'r')

        c = (slo - shi) / (ylo - yhi)
        print('c', c)
        alo = ylo - slo / c
        ahi = yhi - shi / c

        # A estimate is usually good, and both estimates should be close.
        # Exception is when we've got a very steep function, so that the
        # left and right edges are too different. In this case, we should
        # zoom in on where the action is
        # Detect if alo / ahi is far from 1, but avoid dividing by zero.
        print(alo, ahi)
        if alo == 0 or ahi == 0:
            # Difference
            d = np.abs(alo - ahi)
            if alo != ahi and d > 1:
                print(f'Too steep! Difference is {d}')
                print(slo, shi)
                return None
        else:
            # Ratio
            r = np.abs((1 - alo / ahi))
            print(f'Ratio {r}')
            if r > 1:
                print(f'Too steep! r is {r}')
                print(slo, shi)
                return None

        # If decaying, left-most derivative least affected by noise
        if c < 0:
            a = alo
            b = (ylo - alo) * np.exp(-c * xlo)
        else:
            a = ahi
            b = (yhi - ahi) * np.exp(-c * xhi)

        #print(f'A {at:.5f} {alo:+.5e} {ahi:+.5e}')
        #print(f'B {bt:.5f} {b:+.5e}')
        #print(f'C {ct:.3f} {c:.3f}')

        return a, b, c

    ret = inner(x, y)
    print(ret)
    if ret is None:
        print('Warning: repeating initial estimation.'
              ' Function too steep?')
        while ret is None:
            i = len(x) // 2
            qlo, qhi = np.abs(y[0] - y[i]), np.abs(y[i] - y[-1])
            x, y = (x[i:], y[i:]) if qhi > qlo else (x[:-i], y[:-i])
            ret = inner(x, y)

    return ret


def estimate(t, v):
    """
    Fits an exponential ``a + b * exp(c * (t - t[0]))`` to the time series
    ``(t, v)``, returning ``(a, b, c)``

    Example:

        t = ...

    Note that the returned exponentials are defined on ``t - t[0]`` rather
    than ``t``. With this convention, we can analyse segments of a time series
    (i.e. from t=1000 to t=1010) without getting excessively large values of
    ``b``, while at the same time returning a ``b`` that is more likely to
    correspond to a process magnitude (rather than a time-shift).
    """
    # Zero t
    t0 = t[0]
    t = t - t0

    # Attempt untransformed
    a, b, c = estimate_initial(t, v)
    if c == 0:
        return a, b, c

    # Attempt transformed
    rv = (v[-1] - v[0])
    x, y = t / t[-1], (v - v[0]) / rv
    ax, bx, cx = estimate_initial(x, y)
    at = v[0] + ax * rv
    bt = bx * rv
    ct = cx / t[-1]


    from scipy.optimize import minimize as fmin
    f = lambda p: np.sum((p[0] + p[1] * np.exp(p[2] * t) - v)**2)
    r = fmin(f, (a, b, c))
    p = r.x


    fig = plt.figure(figsize=(8, 12))
    fig.subplots_adjust(0.08, 0.08, 0.99, 0.95)
    ax0 = fig.add_subplot(2, 1, 1)
    ax0.set_title('Original')
    ax0.plot(t + t0, v)
    ax0.plot(t + t0, a + b * np.exp(c * t), '-')
    ax0.plot(t + t0, at + bt * np.exp(ct * t), '--')
    ax0.plot(t + t0, p[0] + p[1] * np.exp(p[2] * t), ':')

    ax1 = fig.add_subplot(2, 1, 2)
    ax1.set_title('Transformed')
    ax1.plot(x, y)
    ax1.plot(x, ax + bx * np.exp(cx * x), '--', color='tab:green')

    print(f'Init:   {a:+.5e} {b:+.5e} {c:+.5e}')
    print(f'Init,t: {at:+.5e} {bt:+.5e} {ct:+.5e}')
    print(f'Opt:    {p[0]:+.5e} {p[1]:+.5e} {p[2]:+.5e}')
    #print(f'Opt,t:  {a:+.5e} {b:+.5e} {c:+.5e}')

    #ac, bc, cc = np.mean([[a, b, c], [at, bt, ct]], axis=0)
    #print(np.mean([[a, b, c], [at, bt, ct]], axis=0))

    #ax.plot(x, ac + bc * np.exp(cc * x), ':')
    plt.show()


    return p


class TestTest(unittest.TestCase):
    """
    Teeeeest
    """

    def test_o(self):




        n = 3000
        # TODO: Check equal size, minimum size, etc

        x = np.linspace(1000, 1002, n)
        at, bt, ct = 3, -1e3, -30
        yt = at + bt * np.exp(ct * (x - 1000))

        s = 0.01 * (np.max(yt) - np.min(yt))
        y = yt + np.random.normal(0, s, size=yt.shape)




        # Transform
        a, b, c = estimate(x, y)
        print(f'True:   {at:+.5e} {bt:+.5e} {ct:+.5e}')

        '''
        if True:
            fig = plt.figure(figsize=(8, 5))
            fig.subplots_adjust(0.08, 0.08, 0.99, 0.95)
            ax = fig.add_subplot()
            ax.plot(x, y)
            ax.plot(x, yt)
            ax.plot(x, a + b * np.exp(c * (x - x[0])), '--')
            plt.show()
        '''





if __name__ == '__main__':
    unittest.main()
