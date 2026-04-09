#
# Array and time series vetting
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import numpy as np


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


def vet_series(t, v):
    """
    Assures that ``(t, v)`` is a valid time series.

    Checks that

    - ``t`` and ``v`` are (or can be converted to) 1d numpy arrays.
    - ``t`` and ``v`` are the same size
    - ``t`` is strictly increasing (or has length 0 or 1)

    Returns ``(t, v)`` which are either the original ``t`` and ``v`` (if
    already 1-dimensional numpy arrays) or newly created 1d arrays or views.
    """
    t, m = _vet_array(t)
    v, n = _vet_array(v)
    if m != n:
        raise ValueError(
            f'Both arrays in series must have same length, got {m} and {n}.')
    if n > 1 and np.any(t[1:] <= t[:-1]):
        raise ValueError('The time array must be strictly increasing.')
    return t, v

