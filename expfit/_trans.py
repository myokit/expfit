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
        t = np.asarray(t)
        v = np.asarray(v)
        # Transform
        self._t0 = t[0]
        self._v0 = np.min(v)
        self._rt = t[-1] - self._t0
        self._rv = np.max(v) - self._v0
        if self._rv == 0:
            self._rv = 1
        self.x = (t - self._t0) / self._rt
        self.y = (v - self._v0) / self._rv

    def transform(self, *p):
        """
        Transform parameters ``p = (a, b, c)`` or more generally
        ``p = (a, b_1, c_1, b_2, c_2, ...) to the unit square parameters.

        Can be used as ``transform(a, b, c)`` or ``transform(p)``.
        """
        p = np.asarray(p[0] if len(p) == 1 else p, dtype=float)
        q = np.copy(p)
        q[0] = (p[0] - self._v0) / self._rv
        q[1::2] = p[1::2] / self._rv * np.exp(p[2::2] * self._t0)
        q[2::2] = p[2::2] * self._rt
        return q

    def detransform(self, *q):
        """
        Detransform ``q`` to the original parameter space.

        Can be used as ``detransform(a, b, c)`` or ``detransform(q)``.
        """
        q = np.asarray(q[0] if len(q) == 1 else q, dtype=float)
        p = np.copy(q)
        p[0] = self._v0 + q[0] * self._rv
        p[1::2] = q[1::2] * self._rv * np.exp(-q[2::2] * self._t0 / self._rt)
        p[2::2] = q[2::2] / self._rt
        return p

