#
# Time constants from fits
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import numpy as np

import expfit


def tau1(t, v):
    """
    Fits a single exponential and returns a time constant.
    """
    c = expfit.fit1(t, v)[2]
    if c == 0:
        # Instead of checking sign of zero and returning + or - inf, let numpy
        # handle it (but silently)
        with np.errstate(divide='ignore'):
            return -1 / c
    return -1 / c


'''
def taud2(t, v):
    """
    Fits a double decaying exponential (see :meth:`fit_double_decaying`),
    returning two time constants along with confidence intervals.

    Returns ``(c1, c2, ci1, ci2)`` where ``cix`` is a tuple with the interval.
    """
    p = expfit.fitd2(t, v)
    c1 = p[2]
    c2 = p[4]

    if c1 == 0 or c2 == 0:
        # Instead of checking sign of zero and returning + or - inf, let numpy
        # handle it (but silently)
        with np.errstate(divide='ignore'):
            return -1 / c1, -1 / c2, (np.nan, np.nan), (np.nan, np.nan)

    lo, hi = expfit.ci(t, v, p, 2, constraint=_decaying)
    ci1 = np.sort(np.array((lo[2], hi[2])))
    lo, hi = expfit.ci(t, v, p, 4, constraint=_decaying)
    ci2 = np.sort(np.array((lo[4], hi[4])))
    return c1, c2, ci1, ci2
'''
