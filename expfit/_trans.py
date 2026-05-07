#
# Data transforms
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import numpy as np


class UnitTransform():
    """
    Transforms a time series ``(t, v)`` onto an approximate unit square,
    assuming that ``v[0]`` and ``v[1]`` are good approximations of the signal's
    extremes.

    Properties: ``x``, ``y``.
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
    Transforms a time series by zooming in on the section containing 90% of the
    ???????.


    """
    def __init__(self, t, v, n_min=10):
        self.x = t
        self.y = v
        self._x0 = 0
        self._rx = 1

        # Calculate cumulative variance, starting at peak
        n = len(v)
        mu = np.mean(v)
        reverse = v[-1] > v[0]

        f = 0.80
        if reverse:
            var = np.cumsum((v[::-1] - mu)**2) / np.arange(1, n + 1)
            rng = np.max(var) - np.min(var)
            if rng == 0:
                return
            var = (var[::-1] - np.min(var)) / rng
            ilo = np.where(var >= 1 - f)[0][0]
            ihi = n
        else:
            var = np.cumsum((v - mu)**2) / np.arange(1, n + 1)
            rng = np.max(var) - np.min(var)
            if rng == 0:
                return
            var = (var - np.min(var)) / rng
            ilo = 0
            ihi = np.where(var >= f)[0][0]

        m = ihi - ilo
        if m >= n_min and m < 0.2 * n:
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

