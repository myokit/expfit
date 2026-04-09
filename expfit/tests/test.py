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

from scipy.optimize import minimize as fmin

import expfit

#from myokit.tests import DIR_DATA, WarningCollector



def _vet_array(x):
    """
    Ensures that ``x`` is a 1d numpy array, returns the array and its length.
    """
    if np.isscalar(x):
        return np.array([x], dtype=float), 1

    # Create array from sequence, or just return t if already an array
    x = np.asarray(x)

    # Ensure dimension of 1, all in first coordinate
    if x.ndim == 1:
        # Already OK
        return x, x.shape[0]
    if x.ndim > 1:
        # Size in other directions must be 1
        n = np.max(x.shape)
        if np.prod(x.shape) != n:
            raise ValueError(
                'Unable to convert to 1d vector of scalar values.')
        return x.reshape((n,)), n
    else:
        # 0-dimensional: cast to empty array
        # To test, these can be created with e.g. np.array(0)
        return np.array([]), 0


def vet_series(t, v, minlen=1):
    """
    Assures that ``(t, v)`` is a valid time series.

    Checks that

    - ``t`` and ``v`` are (or can be converted to) 1d numpy arrays.
    - ``t`` and ``v`` are the same size
    - ``t`` is strictly increasing (or has length 1, if ``minlen=1``)

    Returns ``(t, v)`` which are either the original ``t`` and ``v`` (if
    already 1-dimensional numpy arrays) or newly created 1d arrays or views.
    """
    t, m = _vet_array(t)
    v, n = _vet_array(v)
    if m != n:
        raise ValueError(
            f'Both arrays in series must have same length, got {m} and {n}.')
    if m < minlen:
        raise ValueError(
            f'Series of at least length {minlen} required, got {m}.')
    if n > 1 and np.any(t[1:] <= t[:-1]):
        raise ValueError('The time array must be strictly increasing.')
    return t, v


def least_squares(x, y, vet=True):
    """
    Returns a least squares fit ``(a, b)`` where ``y`` is approximated by
    ``a + b * x``.
    """
    if vet:
        x, y = vet_series(x, y, minlen=2)
    n = len(x)
    mu_x, mu_y = np.mean(x), np.mean(y)
    xx = np.sum(x**2) - n * mu_x**2
    xy = np.sum(x * y) - n * mu_x * mu_y
    b = xy / xx
    return mu_y - b * mu_x, b


def find_linear_segment(x, y, min_length, left=True, vet=True):
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
    if vet:
        x, y = vet_series(x, y, minlen=2)

    n = len(x)
    if n <= min_length:
        return n, *least_squares(x, y, vet=False)

    while n > min_length:
        # Fit a straight line
        a, b = least_squares(x, y, vet=False)

        # Calculate residulas
        r = y - (a + b * x)

        # Calculate R**2 in lag-1 autocorrelation
        q = np.corrcoef(r[1:], r[:-1])[0, 1]**2
        if q < 0.1:
            break

        n = max(n // 2, min_length)
        x, y = (x[:n], y[:n]) if left else (x[-n:], y[-n:])

    return n, a, b


def estimate_initial_single(x, y, axes=None, vet=True):
    """
    Estimate ``a, b, c`` in ``y = a + b * exp(c * x)`` using derivatives
    estimated from mean averages at the sides.

    The method first uses :meth:`find_linear_segment` to find two data
    segments, one at the start of the data and one near the end, which are
    well approximated by a straight line. From these segments it then derives
    ``(x1, y1, dydx1)`` and ``(x2, y2, dydx2)`` (where the derivative is
    approximated by the slope of the straight line). It then estimates c from

        y    = a + b * exp(c * x)
        dydx = c * b * exp(c * x)

        y_1    - y_2    =     b * (exp(c * x_1) - exp(c * x_2))
        dydx_1 - dydx_2 = c * b * (exp(c * x_1) - exp(c * x_2))
        c = (dydx_1 - dydx_2) / (y_1 - y_2)

    Either segment can then be used to derive ``a`` and ``b``, from

        a = y_i - dydx_i / c
        b = (y_i - a) / np.exp(c * x_i)

    To pick a segment, the method assumes that the side with the steepest slope
    will have the best signal to noise ratio.

    If a matplot ``axes`` object is passed in, it will plot the used line
    segments.

    Returns a tuple ``(a, b, c)``.
    """
    if vet:
        x, y = vet_series(x, y, minlen=3)

    n = len(x)
    if n >= 10:
        # Estimate the slope of the left-most and right-most segments of the
        # data. Use a segment that provides a good linear fit.
        m_min = 5
        m = max(m_min, n // 10)
        mlo, olo, s0 = find_linear_segment(x[:m], y[:m], m_min, vet=False)
        mhi, ohi, s1 = find_linear_segment(
            x[-m:], y[-m:], m_min, left=False, vet=False)

        # Isolate segments, get means
        xlo, ylo = x[:mlo], y[:mlo]
        xhi, yhi = x[-mhi:], y[-mhi:]
        x0, x1, y0, y1 = np.mean(xlo), np.mean(xhi), np.mean(ylo), np.mean(yhi)

        # Stop if both slopes are exactly the same, and assume a flat line
        if s0 == s1:
            return np.mean(y), 0, 0

        # TODO: Become clever about e.g. slopes pointing towards it not being
        # an exponential

        # Show segments in plot
        if axes is not None:
            axes.plot(x0, y0, 'ks', zorder=10)
            axes.plot(xlo, olo + s0 * xlo, 'k', zorder=10)
            axes.plot(x1, y1, 'rs', zorder=10)
            axes.plot(xhi, ohi + s1 * xhi, 'r', zorder=10)

    elif n > 2:
        # Not enough points to denoise? Just use extremes.

        # Treat as flat if y does not increase monotonically
        y0, y0b, y1b, y1 = y[0], y[1], y[-2], y[-1]
        if (y0b - y0) * (y1 - y1b) <= 0:
            return np.mean(y), 0, 0
        x0, x0b, x1b, x1 = x[0], x[1], x[-2], x[-1]
        s0, s1 = (y0b - y0) / (x0b - x0), (y1 - y1b) / (x1b - x1)
    else:
        raise ValueError('At least 3 points are required')

    # Estimate c, a, and b
    c = (s0 - s1) / (y0 - y1)
    if c < 0:
        a = y0 - s0 / c
        b = (y0 - a) * np.exp(-c * x0)
    else:
        a = y1 - s1 / c
        b = (y1 - a) * np.exp(-c * x1)
    return a, b, c


def rmse_single(x, y, a, b, c):
    """ Returns the RMSE between ``y`` and ``a + b * exp(c * x)``. """
    return np.sqrt(np.sum((y - a - b * np.exp(c * x))**2))


def fit_single(t, v, initial=None, plot=False, vet=True):
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
    if vet:
        t, v = vet_series(t, v)

    # Transform to unit square, to avoid overflows
    rt, rv = (t[-1] - t[0]), (v[-1] - v[0])
    if rv == 0:
        rv = 1
    x, y = (t - t[0]) / rt, (v - v[0]) / rv

    if plot:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 9))
        fig.subplots_adjust(0.08, 0.06, 0.995, 0.995, wspace=0.25, hspace=0.2)
        ax0 = fig.add_subplot(2, 1, 1)
        ax0.set_xlabel('x')
        ax0.set_ylabel('y')
        ax0.plot(x, y, 'k', alpha=0.25, label='Transformed data')
    else:
        ax0 = None

    # Get an initial estimate
    at0, bt0, ct0 = estimate_initial_single(x, y, ax0, vet=False)

    # Fit, unless it's clearly not exponential
    if abs(bt0) < 1e-100:
        at = at0
        bt = ct = bt0 = ct0 = 0
    else:
        r = fmin(lambda p: rmse_single(x, y, *p), (at0, bt0, ct0))
        at, bt, ct = r.x

    # Detransform obtained parameters
    a = v[0] + at * rv
    b = bt * rv * np.exp(-ct * t[0] / rt)
    c = ct / rt

    if plot:
        a0 = v[0] + at0 * rv
        b0 = bt0 * rv * np.exp(-ct0 * t[0] / rt)
        c0 = ct0 / rt

        print(f'Init: {a0:+.5e} {b0:+.5e} {c0:+.5e}')
        print(f'Fit:  {a:+.5e} {b:+.5e} {c:+.5e}')

        ax0.plot(x, at0 + bt0 * np.exp(ct0 * x), '-', label='Initial guess')
        ax0.plot(x, at + bt * np.exp(ct * x), '--', label='Fit')
        ax0.legend()

        ax1 = fig.add_subplot(2, 2, 3)
        ax1.set_xlabel('x')
        ax1.set_ylabel('Residuals')
        ax1.plot(x, y - (at0 + bt0 * np.exp(ct0 * x)), label='Initial')
        ax1.plot(x, y - (at + bt * np.exp(ct * x)), label='Fit')
        ax1.legend()

        ax2 = fig.add_subplot(2, 2, 4)
        ax2.set_xlabel('t')
        ax2.set_ylabel('v')
        ax2.plot(t, v, 'k', alpha=0.25, label='Untransformed data')
        ax2.plot(t, a0 + b0 * np.exp(c0 * t), '-', label='Initial')
        ax2.plot(t, a + b * np.exp(c * t), '--', label='Fit')
        ax2.legend()

    return a, b, c


class TestSingle(unittest.TestCase):
    """
    Teeeeest
    """
    @classmethod
    def setUpClass(cls):
        # Create in each test and seed!
        cls.r = None

    def single_on_single(self, a, b, c, duration, n, fnoise=0.01, t0=0,
                         maxr=1, maxrmse=None, plot=False):
        """
        Fits an exponential to a signal containing an exponential, and returns
        the ratio ``RMSE(fit, noisy) / RMSE(true, noisy)``.

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
        s = max(fnoise * abs(vt[0] - vt[-1]), 1e-9)
        v = vt + self.r.normal(0, s, size=n)

        af, bf, cf = fit_single(t, v, plot=plot)
        rt = rmse_single(t, v, a, b, c)
        rf = rmse_single(t, v, af, bf, cf)

        if plot:
            print(f'True: {a:+.5e} {b:+.5e} {c:+.5e}')
            print(f'RMSE true: {rt}')
            print(f'RMSE fit:  {rf}')
            print(f'ratio: {rf / rt}')
            import matplotlib.pyplot as plt
            plt.show()

        with self.subTest(a=a, b=b, c=c, duration=duration, n=n, fnoise=fnoise,
                          t0=t0):
            if maxrmse is not None:
                self.assertLess(rf, maxrmse)
            if maxr is None and maxrmse is None:
                raise ValueError('Either maxr or maxrmse must be set')


    def single_on_double(self, a, b, c, d, e, duration=1, n=100, fnoise=0.01,
                         t0=0, maxrmse=1, plot=False):
        """
        Fits a single exponential to a signal containing a double exponential.
        """
        t = np.linspace(t0, t0 + duration, n)
        vt = a + b * np.exp(c * t) + d * np.exp(e * t)
        s = max(fnoise * abs(vt[0] - vt[-1]), 1e-9)
        v = vt + self.r.normal(0, s, size=n)

        af, bf, cf = fit_single(t, v, plot=plot)
        rf = rmse_single(t, v, af, bf, cf)

        if plot:
            print(f'True: a {a:+.5e}')
            print(f'      b {b:+.5e} c {c:+.5e}')
            print(f'      d {d:+.5e} e {e:+.5e}')
            print(f'RMSE fit:  {rf}')
            import matplotlib.pyplot as plt
            plt.show()

        with self.subTest(a=a, b=b, c=c, d=d, e=e, duration=duration, n=n,
                          fnoise=fnoise, t0=t0):
            self.assertLess(rf, maxrmse)

    def single_on_triple(self, a, b, c, d, e, f, g, duration=1, n=100,
                         fnoise=0.01, t0=0, maxrmse=1, plot=False):
        """
        Fits a single exponential to a signal containing a double exponential.
        """
        t = np.linspace(t0, t0 + duration, n)
        vt = a + b * np.exp(c * t) + d * np.exp(e * t) + f * np.exp(g * t)
        s = max(fnoise * abs(vt[0] - vt[-1]), 1e-9)
        v = vt + self.r.normal(0, s, size=n)

        af, bf, cf = fit_single(t, v, plot=plot)
        rf = rmse_single(t, v, af, bf, cf)

        if plot:
            print(f'True: a {a:+.5e}')
            print(f'      b {b:+.5e} c {c:+.5e}')
            print(f'      d {d:+.5e} e {e:+.5e}')
            print(f'      f {f:+.5e} g {g:+.5e}')
            print(f'RMSE fit:  {rf}')
            import matplotlib.pyplot as plt
            plt.show()

        with self.subTest(a=a, b=b, c=c, d=d, e=e, duration=duration, n=n,
                          fnoise=fnoise, t0=t0):
            self.assertLess(rf, maxrmse)


    def test_single_on_single(self):
        # Test single exponentials on single exponential data
        sos = self.single_on_single
        self.r = np.random.default_rng(1)
        plot = False

        # Moderate
        sos(0, -1, 3, 2, 123, plot=plot)
        sos(3e2, 2, 4, 2, 100000, plot=plot)
        sos(5e3, 3, -0.5, 5, 500, plot=plot)
        sos(-1e3, 10, -9, 2, 20, plot=plot)

        # Steep
        sos(5e5, -1, 30, 2, 300, maxr=1.3, plot=plot)
        sos(-1e3, 10, -9, 2, 1000, plot=plot)
        sos(5e5, -1, 15, 2, 10000, maxr=1.01, plot=plot)

        # Almost straight
        sos(3, -1, 0.3, 2, 3000, plot=plot)
        sos(-5e2, +1, 0.03, 2, 3000, plot=plot)
        sos(0, 1, 1e-6, 1, 200, plot=plot)

        # Flat
        sos(1, 0, 3, 1, 200, plot=True)

        # Clean
        sos(0, -1, 3, 2, 123, fnoise=0, maxr=None, maxrmse=0.05, plot=plot)
        sos(3e2, 2, 4, 2, 1000, fnoise=1e-3, maxr=1.1, plot=plot)
        sos(5e3, 3, -0.5, 5, 500, fnoise=1e-2, plot=plot)

        # Noisy
        sos(4, 10, 3, 2, 100, fnoise=0.11, plot=True)
        sos(4, 10, 3, 2, 100, fnoise=0.3, maxr=1.15, plot=plot)
        sos(51, -1, -0.5, 5, 200, fnoise=0.5, plot=plot)
        sos(-10, -2, 9, 2, 600, fnoise=1,  plot=True)

    def test_single_on_double(self):
        # Test single exponentials on single exponential data
        sod = self.single_on_double
        self.r = np.random.default_rng(2)

        # Same direction
        #sod(0, -1, 3, -2, 5, maxrmse=30, plot=True)

    def test_single_on_triple(self):
        # Test single exponentials on single exponential data
        sot = self.single_on_triple
        self.r = np.random.default_rng(2)

        # Opposing direction, over fast
        #sot(0, 3, -60, -3, -10, 0, 0, maxrmse=3, plot=True)


if __name__ == '__main__':
    unittest.main()
