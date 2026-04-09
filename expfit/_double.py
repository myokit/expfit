#
# Single expontial fits
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import numpy as np

from scipy.optimize import minimize as fmin

import expfit


def rmse_double(x, y, a, b, c, d, e):
    """
    Returns the RMSE between ``y`` and ``a + b * exp(c * x) + d * exp(e * x)``.
    """
    return np.sqrt(np.sum((y - a - b * np.exp(c * x))**2))


def fit_double(t, v, plot=False, vet=True):


    return

