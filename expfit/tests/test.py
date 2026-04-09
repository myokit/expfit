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
    ``a + b * x``.
    """
    # TODO: Input check, vector, same size etc.
    n = len(x)
    mu_x, mu_y = np.mean(x), np.mean(y)
    xx = np.sum(x**2) - n * mu_x**2
    xy = np.sum(x * y) - n * mu_x * mu_y
    b = xy / xx
    return mu_y - b * mu_x, b


def find_linear_segment(x, y, min_length, left=True):
    """
    Reduces the length of a data set ``(x, y)`` until a straight line provides
    a good prediction of points in ``y`` from ``x``, as judged by
    autocorrelation in the residuals.

    By default, the left-most section of the segment is kept after each
    reduction, but this can be changed by setting ``right=True``.

    Returns a tuple ``(n, a, b)`` such that ``y`` is approximated by
    ``a + b * x`` on a segment of length ``n``, at either the left or the right
    of the data.
    """
    n = len(x)
    while n > min_length:
        # Fit a straight line
        a, b = least_squares(x, y)

        # Calculate residulas
        r = y - (a + b * x)

        # Calculate R**2 in lag-1 autocorrelation
        q = np.corrcoef(r[1:], r[:-1])[0, 1]**2
        if q < 0.1:
            break

        n = max(n // 2, min_length)
        x, y = (x[:n], y[:n]) if left else (x[-n:], y[-n:])

    return n, a, b


def estimate_initial_single(x, y):
    """
    Estimate initial ``a, b, c`` in ``y = a + b exp(c x)`` using
    derivatives estimated from mean averages at the sides.

    The method first uses :meth:`find_linear_segment` to find a segment on
    either extreme of the data (initial points and final points) that is
    approximately linear.

    Next, it

        TODO: Write down how the estimate works





    """
    # Estimate the slope of the left-most and right-most segments of the
    # data. Use a segment that provides a good linear fit.
    # generous part of the signal
    n = len(x)
    if len(x) < 10:
        raise ValueError('TODO: Complain array too short')
    m_min = 3
    m = max(m_min, n // 10)
    mlo, olo, slo = find_linear_segment(x[:m], y[:m], m_min)
    mhi, ohi, shi = find_linear_segment(x[-m:], y[-m:], m_min, left=False)

    # Stop if both slopes are (exactly) the same
    if slo == shi:
        print('Warning: initial estimate suggests flat line')
        return np.mean(y), 0, 0

    # Isolate segments
    xlo, ylo = x[:mlo], y[:mlo]
    xhi, yhi = x[-mhi:], y[-mhi:]
    if False:
        fig = plt.figure(figsize=(8, 5))
        fig.subplots_adjust(0.08, 0.08, 0.99, 0.99)
        ax = fig.add_subplot()
        ax.plot(x, y)
        ax.plot(np.mean(xlo), np.mean(ylo), 'ks')
        ax.plot(xlo, olo + slo * xlo, 'k')
        ax.plot(np.mean(xhi), np.mean(yhi), 'rs')
        ax.plot(xhi, ohi + shi * xhi, 'r')

    # Estimate c in a + b exp(c x)
    mu_ylo, mu_yhi = np.mean(ylo), np.mean(yhi)
    c = (slo - shi) / (mu_ylo - mu_yhi)

    # If decaying, left-most derivative least affected by noise, so will
    # give best estimate of a and b
    if c < 0:
        a = mu_ylo - slo / c
        b = (mu_ylo - a) * np.exp(-c * np.mean(xlo))
    else:
        a = mu_yhi - shi / c
        b = (mu_yhi - a) * np.exp(-c * np.mean(xhi))
    return a, b, c


def rmse_single(x, y, a, b, c):
    """ Returns the RMSE between ``y`` and ``a + b * exp(c * x)``. """
    return np.sqrt(np.sum((y - a - b * np.exp(c * x))**2))


def fit_single(t, v):
    """
    Fits an exponential ``a + b * exp(c * (t - t[0]))`` to the time series
    ``(t, v)``, returning ``(a, b, c)``

    Example:

        t = ...


        TODO


    Note that the fitted function does not take time shifts into account: it is
    assumed that the process matched by the exponential starts at t = 0. If
    this assumption does not hold, the value for ``c`` will still be correct,
    but the values for ``a`` and ``b`` will be affected.
    """
    # Transform to unit square, to avoid overflows
    rt, rv = (t[-1] - t[0]), abs(v[-1] - v[0])
    x, y = (t - t[0]) / rt, (v - v[0]) / rv

    # Get an initial estimate
    at0, bt0, ct0 = estimate_initial_single(x, y)

    # Fit
    from scipy.optimize import minimize as fmin
    r = fmin(lambda p: rmse_single(x, y, *p), (at0, bt0, ct0))
    at, bt, ct = r.x

    # Detransform obtained parameters
    a = v[0] + at * rv
    b = bt * rv * np.exp(-ct * t[0] / rt)
    c = ct / rt

    if False:
        fig = plt.figure(figsize=(8, 12))
        fig.subplots_adjust(0.08, 0.08, 0.99, 0.95)
        ax0 = fig.add_subplot(2, 1, 1)
        ax0.plot(x, y, 'k', alpha=0.25)
        ax0.plot(x, at0 + bt0 * np.exp(ct0 * x), '-', label='Initial guess')
        ax0.plot(x, at + bt * np.exp(ct * x), '--', label='Fit')
        ax0.legend()

        ax1 = fig.add_subplot(2, 1, 2)
        ax1.set_title('Residuals')
        ax1.plot(x, y - (at0 + bt0 * np.exp(ct0 * x)))
        ax1.plot(x, y - (at + bt * np.exp(ct * x)))

    if True:
        a0 = v[0] + at0 * rv
        b0 = bt0 * rv * np.exp(-ct0 * t[0] / rt)
        c0 = ct0 / rt
        print(f'Init: {a0:+.5e} {b0:+.5e} {c0:+.5e}')

    return a, b, c


class TestSingle(unittest.TestCase):
    """
    Teeeeest
    """

    def single(self, a, b, c, duration, n, fnoise=0.01, t0=0,
               d1=False, d2=False):
        """
        Fits an exponential and returns the ratio
        ``RMSE(fit, noisy) / RMSE(true, noisy)``.

        Creates a signal ``a + b exp(c t)`` with ``n`` points from ``t0`` to
        ``t0 + duration``, and normally distributed noise with a variance of
        ``fnoise`` times the clea signal's magnitude.

        Then fits a signal, and calculates the RMSEs between (1) the noisy
        signal and the exponential with the input parameters, and (2) the
        noisy signal and the fit. For a perfect fit the ratio Rfit/Rtrue will
        be less than 1, as the fit will incorporate some of the bias introduced
        by the random noise.
        """

        t = np.linspace(t0, t0 + duration, n)
        vt = a + b * np.exp(c * t)
        v = vt + np.random.normal(0, fnoise * abs(vt[0] - vt[-1]), size=n)

        af, bf, cf = fit_single(t, v)
        rt = rmse_single(t, v, a, b, c)
        rf = rmse_single(t, v, af, bf, cf)

        if d1:
            print(f'True: {a:+.5e} {b:+.5e} {c:+.5e}')
            print(f'Fit:  {af:+.5e} {bf:+.5e} {cf:+.5e}')
            print(f'RMSE true: {rt}')
            print(f'RMSE fit:  {rf}')
            print(f'ratio: {rf / rt}')

        if d2:
            fig = plt.figure(figsize=(8, 5))
            fig.subplots_adjust(0.08, 0.08, 0.99, 0.95)
            ax = fig.add_subplot()
            ax.plot(t, v)
            ax.plot(t, vt)
            ax.plot(t, af + bf * np.exp(cf * t), '--')
            plt.show()

        return rf / rt


    def test_o(self):
        #self.single(3, -1, 3e-1, 2, 3000, t0=1000)

        self.assertLess(self.single(3, -1, 3e-1, 2, 3000), 1)



if __name__ == '__main__':
    unittest.main()
