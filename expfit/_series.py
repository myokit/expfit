#
# Time series object
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import numpy as np


class TimeSeries:
    """
    Stores a time series ``(x, y)``, ensuring it is valid and providing
    cached access to common functions.

    Checks that

    - ``x`` and ``y`` are (or can be converted to) 1d numpy arrays.
    - ``x`` and ``y`` are the same size
    - ``x`` and ``y`` are finite
    - ``x`` is strictly increasing (or has length 0 or 1)

    Access to the underlying data can be gained by treating the
    :class:`TimeSeries` like a tuple::

        x, y = time_series

    or through the public properties ``x`` and ``y``.

    Notes:

    1. It is not necessary to create :class:`TimeSeries` manually. Usually
       these are made internally at the start of the fit, and discarded at the
       end of it.
    2. It is also not recommended: For performance, ``x`` and ``y`` are not
       copied unless necessary (e.g. to convert to float). This means that if
       you _do_ create your own :class:`TimeSeries` and pass it in, you can
       break stuff by changing the underlying data between calls to its
       functions.

    """
    def __init__(self, x, y):
        x, m = self._vet(x)
        y, n = self._vet(y)
        if m != n:
            raise ValueError('Both arrays in series must have same length,'
                             f' got {m} and {n}.')

        if n > 1 and np.any(x[1:] <= x[:-1]):
            raise ValueError('The time array must be strictly increasing.')

        self.x = x
        self.y = y
        self._xy = (x, y)

    @classmethod
    def _from_xy(cls, x, y=None):
        """
        Create from typical arguments ``x`` and ``y``, or return if already a
        :class:`TimeSeries`.
        """
        if isinstance(x, cls):
            if y is not None:
                raise ValueError(
                    'If x is a TimeSeries, y must be None.')
            return x
        elif y is None:
            raise ValueError(
                'If x is an array, y cannot be None')
        return cls(x, y)

    def __len__(self):
        return 2

    def __getitem__(self, subscript):
        return self._xy.__getitem__(subscript)

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


class UnitSquaredSeries(TimeSeries):
    """
    Subclass of :class:`TimeSeries` that presents the data transformed to a
    unit cube.
    """
    def __init__(self, x, y):

        # Get scaling factors
        self._x0 = x[0]
        self._rx = x[-1] - self._x0
        self._y0 = np.min(y)
        self._ry = np.max(y) - self._x0
        if self._ry == 0:
            self._ry = 1

        # Initialise
        super().__init__(
            (x - self._x0) / self._rx,
            (y - self._y0) / self._ry)

        # Set known numbers
        self._min = 0
        self._max = 0 if self._ry == 0 else 1

    def transform(self, p):
        """
        Transform parameters ``p = (a, b, c)`` to the unit square parameters.
        """
        a, b, c = p
        p = (a - self._y0) / self._ry
        q = b / self._ry * np.exp(c * self._x0)
        r = c * self._rx
        return np.array((p, q, r))

    def detransform(self, q):
        """
        Detransform unit square parameters to the original ``(a, b, c)`` space.
        """
        p, q, r = q
        a = self._y0 + self._ry * p
        b = q * self._ry * np.exp(-r * self._x0 / self._rx)
        c = r / self._rx
        return np.array((a, b, c))

