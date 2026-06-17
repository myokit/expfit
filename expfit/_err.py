#
# Error functions for exponential fits.
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import numpy as np


def exp(x, p):
    """
    Returns an exponential ``p[0] + p[1] * exp(-x / p[2]) + p[3] * ...``.
    """
    p = np.asarray(p)
    d = len(p)
    if d < 3 or (d - 1) % 2 != 0:
        raise ValueError(f'Invalid number of parameters ({d}).')

    m = (d - 1) // 2
    b = p[1::2].reshape((m, 1))
    c = p[2::2].reshape((m, 1))
    return p[0] + np.sum(b * np.exp(-np.asarray(x) / c), axis=0)


def expc(x, p):
    """
    Returns an exponential ``p[0] + p[1] * exp(p[2] * x) + p[3] * ...``.
    """
    p = np.asarray(p)
    d = len(p)
    if d < 3 or (d - 1) % 2 != 0:
        raise ValueError(f'Invalid number of parameters ({d}).')

    m = (d - 1) // 2
    b = p[1::2].reshape((m, 1))
    c = p[2::2].reshape((m, 1))
    return p[0] + np.sum(b * np.exp(c * x), axis=0)


def rmse(x, y, p):
    """
    Returns the RMSE between ``y`` and an exponential
    ``p[0] + p[1] * exp(-x / p[1]) + p[3] * exp(-x / p[4]) + ...``.

    **Note**: the returned RMSE is the root of the MSE returned by
    :class:`SingleExponentialError` and :class:`MultiExponentialError`
    """
    # Treat `a` separately: this is  more accurate when a == -b
    # for very large a and b (e.g. straight line)
    p = np.copy(p)
    a = p[0]
    p[0] = 0
    return np.sqrt(np.sum((y - a - exp(x, p))**2) / len(x))


def rmsec(x, y, p):
    """
    Returns the RMSE between ``y`` and an exponential
    ``p[0] + p[1] * exp(p[1] * x) + p[3] * exp(p[4] * x) + ...``.

    **Note**: the returned RMSE is the root of the MSE returned by
    :class:`SingleExponentialError` and :class:`MultiExponentialError`
    """
    p = np.copy(p)
    a = p[0]
    p[0] = 0
    return np.sqrt(np.sum((y - a - expc(x, p))**2) / len(x))


class SingleExponentialError():
    """
    Calculates the MSE, Jacobian, and Hessian for a single exponential.

    This error uses the form::

        y = a + b * exp(c * x)

    with parameters ``p = (a, b, c)``.

    Example::

        x = np.linspace(0, 1, 100)
        y = 5 + 3 * np.exp(0.5 * x)
        e = SingleExponentialError(x, y)
        mse, jac, hes = e([1, 2, 3])

    Arguments:

    ``x``, ``y``
        The time series.

    """
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._m = 1 / len(x)

    def __call__(self, p):
        a, b, c = p
        e = np.exp(c * self._x)
        be = b * e
        f = a - self._y + be
        ef = e * f
        mse = self._m * np.sum(f * f)

        # Jacobian
        jac = np.array([
            2 * self._m * np.sum(f),
            2 * self._m * np.sum(ef),
            2 * self._m * np.sum(ef * self._x) * b
        ])

        # Hessian
        ex = e * self._x
        aex = (f + be) * ex
        hes = np.array([
            [2, 2 * self._m * np.sum(e), 2 * b * self._m * np.sum(ex)],
            [0, 2 * self._m * np.sum(e * e), 2 * self._m * np.sum(aex)],
            [0, 0, 2 * self._m * b * np.sum(self._x * aex)],
        ])
        hes[1, 0] = hes[0, 1]
        hes[2, 0] = hes[0, 2]
        hes[2, 1] = hes[1, 2]

        return mse, jac, hes

    def mse(self, p):
        """ Calculate the MSE without Jacobian or Hessian. """
        return self._m * np.sum(
            (p[0] - self._y + p[1] * np.exp(p[2] * self._x))**2)


class MultiExponentialError():
    """
    Calculates the MSE, Jacobian, and Hessian for multiple decaying
    exponentials, with log-transformed parameters.

    This error uses the form::

        y = a + sum(z[j] * exp(b'[j]) * exp(-exp(c'[j]) * x))

    where ``j`` ranges from ``0`` to ``m - 1``. The parameter vector is::

        p = (a, b'[0], c'[0], b'[1], c'[1], ..., b'[m - 1], c'[m -1])

    and ``(b'[j], c'[j])`` pairs are ordered from slow (largest time constant,
    slowest process, smallest ``c'[j]`` value) to fast.

    The array ``z`` is a constant created at construction time, and determines
    the sign of each component. The first ``m_dominant`` terms will have the
    same sign as the dominant term, and indicated by the boolean
    ``dominant_positive``. The remaining ``m - m_dominant`` terms will have the
    opposite sign.

    Compared to the single exponential error, the ``b`` and ``c`` parameters
    are transformed as::

        a' = a
        b' = log(abs(b))
        c' = log(-c) = -log(tau)

    Example::

        x = np.linspace(0, 1, 20)
        y = 5 + 1.1 * np.exp(4 * x)
        e1 = expfit.SingleExponentialError(x, y)
        e2 = expfit.MultiExponentialError(x, y, 1, 1, True)
        m1, j1, h1 = e1([1, 2, -3])
        m2, j2, h2 = e2([1, np.log(2), np.log(3)])

        # The MSE of both errors is the same:
        print(m1 == m2)

        # But the Jacobian of e2 is in log-transformed space
        print(j1 == j2)

    A multi-exponential example::

        x = np.linspace(0, 1, 20)
        y = 5 + 1.1 * np.exp(4 * x) - 0.9 * np.exp(1 * x)
        e = expfit.MultiExponentialError(x, y, npos=1, nneg=1)
        mse, jac, hes = e([1, 2, 3, 4, 5])

    Arguments:

    ``x``, ``y``
        The time series.
    ``m_dominant``
        The number of exponential terms with the same sign as the dominant
        term.
    ``m_opposite``
        The number of exponential terms with the opposite sign to the dominant
        term.
    ``dominant_positive``
        Determines the sign (strictly positive or negative) of the dominant
        exponential term.

    """
    def __init__(self, x, y, m_dominant, m_opposite, dominant_positive):

        m_dom, m_opp = int(m_dominant), int(m_opposite)
        if m_dom < 1:
            raise ValueError(
                'Number of exponential terms with same sign as dominant term'
                ' must be one or greater.')
        if m_opp < 0:
            raise ValueError(
                'Number of exponential terms with opposite sign to the'
                ' dominant term can not be negative.')

        self._x = x
        self._y = y
        self._ni = 1 / len(x)
        self._n2 = 2 * self._ni

        self._m = m_dom + m_opp
        self._np = 1 + 2 * self._m

        self._z = np.ones(self._m)
        if dominant_positive:
            self._z[m_dom:] = -1
        else:
            self._z[:m_dom] = -1

    def __call__(self, p):
        if len(p) != self._np:
            raise ValueError(f'Expecting {self._np} parameters, got {len(p)}.')
        m, d = self._m, self._np

        # Unpack and detransform
        p = np.asarray(p)
        b = (self._z * np.exp(p[1::2])).reshape((m, 1))  # (m, 1)
        c = -np.exp(p[2::2]).reshape((m, 1))             # (m, 1)

        # MSE
        e = np.exp(c * self._x)                  # (m, n)
        be = b * e                               # (m, n)
        f = p[0] - self._y + np.sum(be, axis=0)  # (n, )
        mse = np.sum(f**2) * self._ni

        # Jacobian
        ex = e * self._x  # (m, n)
        bcT = (b * c).T   # (1, m)
        jac = np.zeros(d)
        jac[0] = self._n2 * np.sum(f)
        jac[1::2] = self._n2 * np.sum(f * e, axis=1) * b.T
        jac[2::2] = self._n2 * np.sum(f * ex, axis=1) * bcT

        # Hessian
        fbe = (f + be)        # (m, n)
        fbex = fbe * self._x  # (m, n)
        # aa, ab, ac
        hes = np.zeros((d, d))
        hes[0, 0] = 2
        hes[0, 1::2] = hes[1::2, 0] = self._n2 * np.sum(e, axis=1) * b.T
        hes[0, 2::2] = hes[2::2, 0] = self._n2 * np.sum(ex, axis=1) * bcT
        for i in range(m):
            # bi^2, ci^2, sand bi*ci
            hes[1 + 2 * i, 1 + 2 * i] = \
                self._n2 * np.sum(fbe[i] * e[i]) * b[i, 0]
            hes[2 + 2 * i, 2 + 2 * i] = \
                self._n2 * np.sum((fbex[i] * c[i, 0] + f) * ex[i]) * bcT[0, i]
            hes[1 + 2 * i, 2 + 2 * i] = hes[2 + 2 * i, 1 + 2 * i] = \
                self._n2 * np.sum(fbex[i] * e[i]) * bcT[0, i]
            for j in range(i + 1, m):
                eij = e[i] * e[j]
                eijx = eij * self._x
                seijx = np.sum(eijx)
                # bi*bj, ci*cj, bi*cj, bj*ci
                hes[1 + 2 * i, 1 + 2 * j] = hes[1 + 2 * j, 1 + 2 * i] = \
                    self._n2 * np.sum(eij) * b[i, 0] * b[j, 0]
                hes[2 + 2 * i, 2 + 2 * j] = hes[2 + 2 * j, 2 + 2 * i] = \
                    self._n2 * np.sum(eijx * self._x) * bcT[0, i] * bcT[0, j]
                hes[1 + 2 * i, 2 + 2 * j] = hes[2 + 2 * j, 1 + 2 * i] = \
                    self._n2 * seijx * bcT[0, j] * b[i, 0]
                hes[2 + 2 * i, 1 + 2 * j] = hes[1 + 2 * j, 2 + 2 * i] = \
                    self._n2 * seijx * bcT[0, i] * b[j, 0]

        return mse, jac, hes

    def mse(self, p):
        """ Calculate the MSE without Jacobian or Hessian. """
        if len(p) != self._np:
            raise ValueError(f'Expecting {self._np} parameters, got {len(p)}.')

        p = np.asarray(p)
        b = (self._z * np.exp(p[1::2])).reshape((self._m, 1))
        c = -np.exp(p[2::2]).reshape((self._m, 1))
        return self._ni * np.sum(
            (p[0] - self._y + np.sum(b * np.exp(c * self._x), axis=0))**2)

    def transform(self, p, tau=False):
        """ Converts ``p`` from c or tau form to log-transformed. """
        if len(p) != self._np:
            raise ValueError(f'Expecting {self._np} parameters, got {len(p)}.')
        q = np.array(p, dtype=float, copy=True)
        q[1::2] = np.log(self._z * p[1::2])
        if tau:
            q[2::2] = -np.log(p[2::2])
        else:
            q[2::2] = np.log(-p[2::2])
        return q

    def detransform(self, q, tau=False):
        """ Converts ``q`` from log-transformed to tau-form. """
        if len(q) != self._np:
            raise ValueError(f'Expecting {self._np} parameters, got {len(q)}.')
        p = np.array(q, dtype=float, copy=True)
        p[1::2] = self._z * np.exp(q[1::2])
        if tau:
            p[2::2] = np.exp(-q[2::2])
        else:
            p[2::2] = -np.exp(q[2::2])
        return p


class TauFormError():
    """
    Calculates the MSE, Jacobian, and Hessian for multiple decaying
    exponentials, using time constant as parameters.

    This error uses the form::

        y = a + sum(b[i] * exp(-np.exp(c[i]) * x))

    Arguments:

    ``x``, ``y``
        The time series

    """
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._ni = 1 / len(x)
        self._n2 = 2 * self._ni

    def __call__(self, p):
        d = len(p)
        if d < 3 or (d - 1) % 2 != 0:
            raise ValueError(f'Invalid number of parameters ({d}).')
        m = (d - 1) // 2

        # Unpack
        p = np.asarray(p)
        a = p[0]
        b = p[1::2].reshape((m, 1))       # (m, 1)
        c = -1 / p[2::2].reshape((m, 1))  # (m, 1)

        # MSE
        e = np.exp(c * self._x)               # (m, n)  e^(cx)
        be = b * e                            # (m, n) be^(cx)
        f = a - self._y + np.sum(be, axis=0)  # (n, ) a - y + sum_j(be^(cx))
        mse = self._ni * np.sum(f**2)

        # Jacobian
        c2 = c * c
        bc2 = b * c2
        ex = e * self._x
        jac = np.zeros(d)
        jac[0] = self._n2 * np.sum(f)
        jac[1::2] = self._n2 * np.sum(f * e, axis=1)
        jac[2::2] = self._n2 * np.sum(f * ex, axis=1) * bc2.T

        # Hessian
        fbe = f + be          # (m, n)
        fbex = fbe * self._x  # (m, n)
        # aa, ab, ac
        hes = np.zeros((d, d))
        hes[0, 0] = 2
        hes[0, 1::2] = hes[1::2, 0] = self._n2 * np.sum(e, axis=1)
        hes[0, 2::2] = hes[2::2, 0] = self._n2 * np.sum(ex, axis=1) * bc2.T
        for i in range(m):
            # bi^2, ci^2, and bi*ci
            hes[1 + 2 * i, 1 + 2 * i] = self._n2 * np.sum(e[i]**2)
            hes[2 + 2 * i, 2 + 2 * i] = self._n2 * np.sum(
                (fbex[i] * c[i, 0] + 2 * f) * ex[i]) * bc2[i, 0] * c[i, 0]
            hes[1 + 2 * i, 2 + 2 * i] = hes[2 + 2 * i, 1 + 2 * i] = \
                self._n2 * np.sum(fbex[i] * e[i]) * c2[i, 0]
            for j in range(i + 1, m):
                eij = e[i] * e[j]
                eijx = eij * self._x
                seijx = np.sum(eijx)
                # bi*bj, ci*cj, bi*cj, bj*ci
                hes[1 + 2 * i, 1 + 2 * j] = hes[1 + 2 * j, 1 + 2 * i] = \
                    self._n2 * np.sum(eij)
                hes[2 + 2 * i, 2 + 2 * j] = hes[2 + 2 * j, 2 + 2 * i] = \
                    self._n2 * np.sum(eijx * self._x) * bc2[i, 0] * bc2[j, 0]
                hes[1 + 2 * i, 2 + 2 * j] = hes[2 + 2 * j, 1 + 2 * i] = \
                    self._n2 * seijx * bc2[j, 0]
                hes[2 + 2 * i, 1 + 2 * j] = hes[1 + 2 * j, 2 + 2 * i] = \
                    self._n2 * seijx * bc2[i, 0]

        return mse, jac, hes

    def mse(self, p):
        """ Calculate the MSE without Jacobian or Hessian. """
        d = len(p)
        if d < 3 or (d - 1) % 2 != 0:
            raise ValueError(f'Invalid number of parameters ({d}).')
        m = (d - 1) // 2
        p = np.asarray(p)
        return self._ni * np.sum((p[0] - self._y + np.sum(
            p[1::2].reshape((m, 1)) * np.exp(
                -1 / p[2::2].reshape((m, 1)) * self._x), axis=0))**2)


class ErrorWithFixedParameter():
    """
    Wraps around an error class and turns one parameter into a constant.

    This is used in profiling methods.

    Arguments:

    ``error``
        The error to wrap.
    ``p``
        The best solution
    ``ifix``
        The index in the parameter vector of the parameter to hold fixed.

    """
    def __init__(self, error, p, ifix):
        self._e = error
        self._p = np.copy(p)
        self._i = int(ifix)

    def __call__(self, p):
        self._p[:self._i] = p[:self._i]
        self._p[self._i + 1:] = p[self._i:]
        m, j, h = self._e(self._p)
        j = np.delete(j, self._i, 0)
        h = np.delete(np.delete(h, self._i, axis=0), self._i, axis=1)
        return m, j, h

