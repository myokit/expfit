#
# Single expontial fits
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import numpy as np

from scipy.optimize import minimize as fmin

import expfit


def least_squares(x, y, vet=True):
    """
    Returns a least squares fit ``(a, b)`` where ``y`` is approximated by
    ``a + b * x``.
    """
    if vet:
        x, y = expfit.vet_series(x, y)
    n = len(x)
    if n < 2:
        raise ValueError('At least 2 points are required')

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
        x, y = expfit.vet_series(x, y)
    n = len(x)
    if n < 2:
        raise ValueError('At least 2 points are required')

    # Fit a straight line
    a, b = least_squares(x, y, vet=False)

    while n > min_length:

        # Calculate residulas
        r = y - (a + b * x)

        # Calculate R**2 in lag-1 autocorrelation
        q = np.corrcoef(r[1:], r[:-1])[0, 1]**2
        if q < 0.1:
            break

        n = max(n // 2, min_length)
        x, y = (x[:n], y[:n]) if left else (x[-n:], y[-n:])
        a, b = least_squares(x, y, vet=False)

    return x, y, a, b


def estimate_initial_single(x, y, azero=False, axes=None, vet=True):
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

    Arguments:

    ``x``
        A time vector
    ``y``
        The corresponding values
    ``azero=False``
        This can be set to true to force ``a=0``
    ``axes=None``
        Pass in matplotlib Axes to obtain a plot of the selected segments and
        estimates slopes.
    ``vet=True``
        This can be used to disable checks on the dimensions of ``x`` and'
        ``y``.

    If a matplot ``axes`` object is passed in, it will plot the used line
    segments.

    Returns a tuple ``(a, b, c)``.
    """
    if vet:
        x, y = expfit.vet_series(x, y)
    n = len(x)
    if n < 3:
        raise ValueError('At least 3 points are required')

    # Obtain segments to base estimates on
    if n >= 10:
        # Estimate the slope of the left-most and right-most segments of the
        # data. Use a segment that provides a good linear fit.
        m_min = 5
        if n >= 1000:
            m = max(m_min, n // 10)
        elif n >= 100:
            m = max(m_min, n // 3)
        else:
            m = max(m_min, n // 2)

        xlo, ylo, olo, s0 = find_linear_segment(
            x[:m], y[:m], m_min, vet=False)
        xhi, yhi, ohi, s1 = find_linear_segment(
            x[-m:], y[-m:], m_min, left=False, vet=False)

    else:
        # Use left half and right half, rounding up
        m = (n + 1) // 2
        xlo, ylo, xhi, yhi = x[:m], y[:m], x[-m:], y[-m:]
        x0, x1, y0, y1 = np.mean(xlo), np.mean(xhi), np.mean(ylo), np.mean(yhi)
        olo, s0 = least_squares(xlo, ylo, vet=False)
        ohi, s1 = least_squares(xhi, yhi, vet=False)

    # Use means of selected segments
    x0, x1, y0, y1 = np.mean(xlo), np.mean(xhi), np.mean(ylo), np.mean(yhi)

    # Show segments in plot
    if axes is not None:  # pragma: no cover
        axes.plot(x0, y0, 'ks', zorder=10)
        axes.plot(xlo, olo + s0 * xlo, 'k', zorder=10)
        axes.plot(x1, y1, 'rs', zorder=10)
        axes.plot(xhi, ohi + s1 * xhi, 'r', zorder=10)

    # TODO: Become clever about e.g. slopes pointing towards it not being
    # an exponential?

    # Stop if both slopes are exactly the same, and assume a flat line
    if s0 == s1:
        return np.mean(y), 0, 0

    # Estimate c, a, and b
    c = (s0 - s1) / (y0 - y1)
    if c < 0:
        a = 0 if azero else y0 - s0 / c
        b = (y0 - a) * np.exp(-c * x0)
    else:
        a = 0 if azero else y1 - s1 / c
        b = (y1 - a) * np.exp(-c * x1)

    # Very low b? Then assume flat line
    if abs(b) < 1e-100:
        b = c = 0

    return a, b, c


def rmse_single(x, y, a, b, c):
    """ Returns the RMSE between ``y`` and ``a + b * exp(c * x)``. """
    return np.sqrt(np.sum((y - a - b * np.exp(c * x))**2))


def fit_single(t, v, plot=False, vet=True):
    """
    Fits an exponential ``a + b * exp(c * t)`` to the time series ``(t, v)``,
    returning ``(a, b, c)``

    Example::

        t = np.linspace(0, 1, 100)
        v = 3 - 2 * np.exp(4 * t) + np.random.normal(0, 1)
        a, b, c = expfit.fit_single(t, v)
        print(a, b, c)

    """
    if vet:
        t, v = expfit.vet_series(t, v)

    # Transform to unit square, to avoid overflows
    rt, rv = (t[-1] - t[0]), (v[-1] - v[0])
    if rv == 0:
        rv = 1
    x, y = (t - t[0]) / rt, (v - v[0]) / rv

    # Create initial plot
    if plot:  # pragma: no cover
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 9))
        fig.subplots_adjust(0.095, 0.06, 0.995, 0.995, wspace=0.3, hspace=0.2)
        ax0 = fig.add_subplot(2, 1, 1)
        ax0.set_xlabel('x')
        ax0.set_ylabel('y')
        code, color = ('-', '#92cc92') if len(x) > 10 else ('x-', 'tab:green')
        ax0.plot(x, y, code, color=color, label='Transformed data')
    else:
        ax0 = None

    # Get an initial estimate
    at0, bt0, ct0 = estimate_initial_single(x, y, axes=ax0, vet=False)

    # Fit
    with np.errstate(all='ignore'):
        r = fmin(lambda p: rmse_single(x, y, *p), (at0, bt0, ct0))
    at, bt, ct = r.x

    # Detransform obtained parameters
    a = v[0] + at * rv
    b = bt * rv * np.exp(-ct * t[0] / rt)
    c = ct / rt

    if plot:  # pragma: no cover
        a0 = v[0] + at0 * rv
        b0 = bt0 * rv * np.exp(-ct0 * t[0] / rt)
        c0 = ct0 / rt

        print()
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
        ax2.plot(t, v, code, color=color, label='Untransformed data')
        ax2.plot(t, a0 + b0 * np.exp(c0 * t), '-', label='Initial')
        ax2.plot(t, a + b * np.exp(c * t), '--', label='Fit')
        ax2.legend()

    return a, b, c

