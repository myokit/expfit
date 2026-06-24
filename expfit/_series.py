#
# Time series object
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import numpy as np


class TimeSeries:
    """
    Stores a time series ``(t, v)``, ensuring it is valid and providing
    cached access to common functions.

    Checks that

    - ``t`` and ``v`` are (or can be converted to) 1d numpy arrays.
    - ``t`` and ``v`` are the same size
    - ``t`` and ``v`` are finite
    - ``t`` is strictly increasing (or has length 0 or 1)

    Access to the underlying data can be gained by treating the
    :class:`TimeSeries` like a tuple::

        t, v = time_series

    Notes:

    1. It is not necessary to create :class:`TimeSeries` manually. Usually
       these are made internally at the start of the fit, and discarded at the
       end of it.
    2. It is also not recommended: For performance, ``t`` and ``v`` are not
       copied unless necessary (e.g. to convert to float). This means that if
       you _do_ create your own :class:`TimeSeries` and pass it in, you can
       break stuff by changing the underlying data between calls to its
       functions.

    """
    def __init__(self, t, v):
        t, m = self._vet(t)
        v, n = self._vet(v)
        if m != n:
            raise ValueError('Both arrays in series must have same length,'
                             f' got {m} and {n}.')

        if n > 1 and np.any(t[1:] <= t[:-1]):
            raise ValueError('The time array must be strictly increasing.')

        self._tv = t, v
        self._min = None
        self._max = None
        self._mean = None
        self._std = None

    @classmethod
    def _from_tv(cls, t, v=None):
        """
        Create from typical arguments t and v, or return if already a
        :class:`TimeSeries`.
        """
        if isinstance(t, cls):
            if v is not None:
                raise ValueError(
                    'TimeSeries given as `t` argument, but `v` is not None.')
            return t
        return cls(t, v)

    def __len__(self):
        return 2

    def __getitem__(self, subscript):
        return self._tv.__getitem__(subscript)

    def _vet(self, x):
        """ Returns ``x`` as a 1d numpy array, plus its length. """
        if np.isscalar(x):
            return np.array([x], dtype=float), 1

        # Create array from sequence, or just return it if already an array
        x = np.asarray(x, dtype=float)

        # Ensure is finite
        if not np.all(np.isfinite(x)):
            raise ValueError(
                'Arrays used in time series must contain finite values.')

        # Ensure dimension of 1, all in first coordinate
        if x.ndim == 1:
            # Already OK
            return x, x.size
        elif x.ndim > 1:
            # Size in other directions must be 1
            n = x.size
            if np.max(x.shape) != n:
                raise ValueError(
                    'Unable to convert array in time series to 1d vector.')
            return x.reshape((n,)), n
        else:
            # 0-dimensional: cast to empty array
            # To test, these can be created with e.g. np.array(0)
            return np.array([], dtype=float), 0

    def max(self):
        """ The maximum value in ``v``. """
        if self._max is None:
            self._max = np.max(self.tv[1])
        return self._max

    def mean(self):
        """ The mean of ``v``. """
        if self._mean is None:
            self._mean = np.mean(self.tv[1])
        return self._mean

    def min(self):
        """ The minimum value in ``v``. """
        if self._min is None:
            self._min = np.min(self.tv[1])
        return self._min

    def std(self):
        """ The uncorrected sample standard deviation of ``v``. """
        if self._std is None:
            self._std = np.std(v, mean=self.mean())
        return self._std




class UnitSquareTransformedTimeSeries(TimeSeries):
    """
    Subclass of :class:`TimeSeries` that presents the data transformed to a
    unit cube.
    """
    def __init__(self, t, v):

        # Get scaling factors
        self._t0 = t[0]
        self._rt = t[-1] - self._t0
        self._v0 = np.min(v)
        self._rv = np.max(v) - self._v0
        if self._rv == 0:
            self._rv = 1

        # Initialise
        super().__init__(
            (t - self._t0) / self._rt,
            (v - self._v0) / self._rv)

        # Set known numbers
        self._min = 0
        self._max = 0 if self._rv == 0 else 1

    def transform(self, *p):
        """
        Transform parameters ``p = (a, b, tau)`` or more generally
        ``p = (a, b_1, tau_1, b_2, tau_2, ...) to the unit square parameters.

        Can be used as ``transform(a, b, tau)`` or ``transform(p)``.
        """
        p = np.asarray(p[0] if len(p) == 1 else p, dtype=float)
        q = np.copy(p)
        q[0] = (p[0] - self._v0) / self._rv
        q[1::2] = p[1::2] / self._rv * np.exp(-self._t0 / p[2::2])
        q[2::2] = p[2::2] / self._rt
        return q

    def detransform(self, *q):
        """
        Detransform ``q`` to the original parameter space.

        Can be used as ``detransform(a, b, tau)`` or ``detransform(q)``.
        """
        q = np.asarray(q[0] if len(q) == 1 else q, dtype=float)
        p = np.copy(q)
        p[0] = self._v0 + self._rv * q[0]
        p[1::2] = q[1::2] * self._rv * np.exp(self._t0 / (self._rt * q[2::2]))
        p[2::2] = q[2::2] * self._rt
        return p
