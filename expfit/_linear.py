#
# Linear fitting methods, form the basis of the single-exponential fits
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import numpy as np

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

    Returns a tuple ``(xx, yy, a, b)`` such that ``yy`` is approximated by
    ``a + b * xx`` on a segment ``(xx, yy)`` of the original time series
    ``(x, y)`` at either the left or right side of the data.
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

