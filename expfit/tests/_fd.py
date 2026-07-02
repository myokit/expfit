#!/usr/bin/env python3
#
# Test module for expfit
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import expfit

import numpy as np


class FDError:
    """
    Finite-difference error with jacobian and hessian, based on a subclassable
    method :meth:`mse`.
    """
    def __init__(self, x, y):
        self._x, self._y = expfit.TimeSeries._from_xy(x, y)
        self._ni = 1 / len(self._x)

    def mse(self, p):
        """ Calculates an MSE, for a single exponential in c-form. """
        m = (len(p) - 1) // 2
        p = np.asarray(p)
        b = p[1::2].reshape((m, 1))
        c = p[2::2].reshape((m, 1))
        return self._ni * np.sum(
            (p[0] - self._y + np.sum(b * np.exp(c * self._x), axis=0))**2)

    def mse_jac(self, p, dp=1e-6):
        """ Multi-exponential MSE plus jacobian by finite differences. """
        e = self.mse(p)
        jac = np.zeros(len(p))
        p = np.array(p, dtype=float)
        for i in range(len(p)):
            q = np.copy(p)
            q[i] += dp
            jac[i] = (self.mse(q) - e) / dp
        return e, jac

    def __call__(self, p, dp=1e-6):
        """
        Multi-exponential MSE, Jacobian, and Hessian by finite differences.
        """
        d = len(p)
        mse, jac = self.mse_jac(p, dp)
        hes = np.zeros((d, d))
        p = np.array(p, dtype=float)
        for i in range(len(p)):
            q = np.copy(p)
            q[i] += dp
            hes[i] = (self.mse_jac(q, dp)[1] - jac) / dp
        return mse, jac, hes


class FDErrorMulti(FDError):
    """ Finite differences for log-transformed multi-exponential form. """
    def __init__(self, x, y, m_dom, m_opp, dom_pos):
        super().__init__(x, y)
        self._z = np.ones(m_dom + m_opp) * (1 if dom_pos else -1)
        self._z[m_dom:] *= -1

    def mse(self, p):
        """ Calculates an MSE, for a single exponential in c-form. """
        p = np.copy(p)
        p[1::2] = (self._z * np.exp(p[1::2]))
        p[2::2] = -np.exp(p[2::2])
        return super().mse(p)


class FDErrorTau(FDError):
    """ Finite differences for tau form. """
    def mse(self, p):
        """ Calculates an MSE, for a single exponential in c-form. """
        p = np.copy(p)
        p[2::2] = -1 / p[2::2]
        return super().mse(p)

