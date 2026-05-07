#
# Data transforms
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import numpy as np


class UnitSquareTransform():
    """
    Transforms a time series ``(t, v)`` onto an approximate unit square,
    assuming that ``v[0]`` and ``v[-1]`` are good approximations of the
    signal's extrema.

    Public properties: ``x``, ``y``.
    """
    def __init__(self, t, v, n_min=10):
        # Transform
        self._t0 = t[0]
        self._v0 = v[0]
        self._rt = t[-1] - self._t0
        self._rv = v[-1] - self._v0
        if self._rv == 0:
            self._rv = 1
        self.x = (t - self._t0) / self._rt
        self.y = (v - self._v0) / self._rv

    def transform(self, a, b, c):
        """
        Transform ``a``, ``b``, and ``c`` to the unit square parameters.
        """
        p = (a - self._v0) / self._rv
        q = b / self._rv * np.exp(c * self._t0)
        r = c * self._rt
        return p, q, r

    def detransform(self, p, q, r):
        """
        Detransform ``p``, ``q``, and ``r`` to the original parameters.
        """
        a = self._v0 + p * self._rv
        b = q * self._rv * np.exp(-r * self._t0 / self._rt)
        c = r / self._rt
        return a, b, c


class ZoomTransform():
    """
    Transforms a time series to try and zoom in on the action, for very steep
    exponentials.

    Tests wether there is a segment at the start or end of the signal, in which
    the range of ``v`` exceeds ``r_factor`` times the range outside this
    segment. If this segment exists, and has length greather than ``n_min``,
    the transform returns only that segment. If no such segment is found, it
    returns the full signal.

    Public properties: ``x``, ``y``, ``ibounds=None``.
    """
    def __init__(self, t, v, r_factor=20, n_min=10):
        self.x = t
        self.y = v
        self.ibounds = None
        self._x0 = 0
        self._rx = 1

        # Try zooming in on left or right of signal
        ilo = ihi = None
        n = len(v)
        m = n // 2
        s1, s2 = v[:m], v[m:]
        r1, r2 = np.max(s1) - np.min(s1), np.max(s2) - np.min(s2)
        if r2 != 0 and r1 / r2 > r_factor:
            while r2 != 0 and r1 / r2 > r_factor and m > 1:
                m = max(m // 2, 1)
                s1, s2 = v[:m], v[m:]
                r1, r2 = np.max(s1) - np.min(s1), np.max(s2) - np.min(s2)
            ilo, ihi = 0, m
        elif r1 != 0 and r2 / r1 > r_factor:
            while r1 != 0 and r2 / r1 > r_factor and m > 1:
                m = max(m // 2, 1)
                s1, s2 = v[:-m], v[-m:]
                r1, r2 = np.max(s1) - np.min(s1), np.max(s2) - np.min(s2)
            ilo, ihi = n - m, n

        # Concentrated in too small an area? Treat as distortion and ignore
        if ilo is not None and m < n_min:
            # print('Rejecting selection: too small')
            ilo = ihi = None

        # Apply
        if ilo is not None:
            self.ibounds = (ilo, ihi)
            self._x0 = self.x[ilo]
            self._rx = self.x[ihi - 1] - self._x0
            self.x = (self.x[ilo:ihi] - self._x0) / self._rx
            self.y = self.y[ilo:ihi]

    def transform(self, a, b, c):
        """
        Transform ``a``, ``b``, and ``c`` to the unit square parameters.
        """
        if self._rx != 1:
            b = b * np.exp(c * self._x0)
            c = c * self._rx
        return a, b, c

    def detransform(self, p, q, r):
        """
        Detransform ``p``, ``q``, and ``r`` to the original parameters.
        """
        if self._rx != 1:
            q = q * np.exp(-r * self._x0 / self._rx)
            r = r / self._rx
        return p, q, r

    def detransform_series(self, x, y):
        """
        Detransform time series ``(x, y)`` to the original space.
        """
        if self._rx != 1:
            return self._rx * x + self._x0, y
        return x, y


class NoTransform():  # pragma: no cover
    """
    Doesn't.
    """
    def __init__(self, t, v):
        self.x = t
        self.y = v

    def transform(self, a, b, c):
        return a, b, c

    def detransform(self, p, q, r):
        return p, q, r

    def detransform_series(self, x, y):
        return x, y

