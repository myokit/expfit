#
# Error functions for exponential fits.
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import numpy as np


def exp(x, p):
    """
    Returns an exponential ``p[0] + p[1] * exp(p[2] * x) + p[3] * ...``.
    """
    p = np.asarray(p)
    d = len(p)
    assert (d - 1) % 2 == 0
    m = (d - 1) // 2
    b = p[1::2].reshape((m, 1))
    c = p[2::2].reshape((m, 1))
    return p[0] + np.sum(b * np.exp(c * x), axis=0)


def rmse(x, y, p):
    """
    Returns the RMSE between ``y`` and an exponential
    ``p[0] + p[1] * exp(p[1] * x) + p[3] * exp(p[4] * x) + ...``.

    **Note**: the returned RMSE is the root of the MSE returned by
    :class:`SingleExponentialError` and :class:`MultiExponentialError`
    """
    # Treat `a` separately: this is  more accurate when a == -b
    # for very large a and b (e.g. straight line)
    p = np.array(p, copy=True)
    a = p[0]
    p[0] = 0
    return np.sqrt(np.sum((y - a - exp(x, p))**2) / len(x))


class SingleExponentialError():
    """
    Callable class returning the MSE and its Jacobian and Hessian for a single
    exponential ``y = a + b * exp(c * x)`` fit with parameters
    ``p = (a, b, c)``.
    """
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._m = 1 / len(x)

    def __call__(self, p):
        a, b, c = p
        e = np.exp(c * self._x)
        f = a - self._y + b * e
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
        aex = (a - self._y + 2 * b * e) * ex
        hes = np.array([
            [2, 2 * self._m * np.sum(e), 2 * b * self._m * np.sum(ex)],
            [0, 2 * self._m * np.sum(e * e), 2 * self._m * np.sum(aex)],
            [0, 0, 2 * self._m * b * np.sum(self._x * aex)],
        ])
        hes[1, 0] = hes[0, 1]
        hes[2, 0] = hes[0, 2]
        hes[2, 1] = hes[1, 2]

        return mse, jac, hes


class MultiExponentialError():
    """
    Callable class returning the MSE and its Jacobian and Hessian for a
    multi-exponential ``y = a + b_i * exp(c_i * x)`` fit with parameters
    ``p = (a, b_1, c_1, b_2, c_2, ...)``.
    """
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._ni = 1 / len(x)
        self._n2 = 2 * self._ni

    def __call__(self, p):
        d = len(p)
        assert (d - 1) % 2 == 0 and d > 1
        m = (d - 1) // 2

        # Unpack
        p = np.asarray(p)
        a = p[0]
        bs = p[1::2].reshape((m, 1))        # (m, 1)
        cs = p[2::2].reshape((m, 1))        # (m, 1)

        # MSE
        es = np.exp(np.outer(cs, self._x))      # (m, n)  e^(cx)
        bes = bs * es                           # (m, n) be^(cx)
        fs = a - self._y + np.sum(bes, axis=0)  # (n, ) a - y + sum_j(be^(cx))
        mse = np.sum(fs**2) * self._ni

        # Jacobian
        jac = np.zeros(d)
        xes = es * self._x
        jac[0] = self._n2 * np.sum(fs)
        jac[1::2] = self._n2 * np.sum(fs * es, axis=1)
        jac[2::2] = self._n2 * np.sum(fs * xes, axis=1) * bs.T

        # Hessian
        hes = np.zeros((d, d))

        # aa, ab, ac
        hes[0, 0] = 2
        hes[0, 1::2] = hes[1::2, 0] = self._n2 * np.sum(es, axis=1)
        hes[0, 2::2] = hes[2::2, 0] = self._n2 * np.sum(xes, axis=1) * bs.T
        for i in range(m):
            # bi^2, ci^2, and bi*ci
            hes[1 + 2 * i, 1 + 2 * i] = self._n2 * np.sum(es[i]**2)
            hes[2 + 2 * i, 2 + 2 * i] = \
                self._n2 * np.sum((fs + bes[i]) * xes[i] * self._x) * bs[i, 0]
            hes[1 + 2 * i, 2 + 2 * i] = hes[2 + 2 * i, 1 + 2 * i] = \
                self._n2 * np.sum((fs + bes[i]) * xes[i])

            for j in range(i + 1, m):
                # bi*bj, ci*cj, bi*cj, bj*ci
                hes[1 + 2 * i, 1 + 2 * j] = hes[1 + 2 * j, 1 + 2 * i] = \
                    self._n2 * np.sum(es[i] * es[j])
                hes[2 + 2 * i, 2 + 2 * j] = hes[2 + 2 * j, 2 + 2 * i] = \
                    self._n2 * np.sum(xes[i] * xes[j]) * bs[i, 0] * bs[j, 0]
                hes[1 + 2 * i, 2 + 2 * j] = hes[2 + 2 * j, 1 + 2 * i] = \
                    self._n2 * np.sum(xes[i] * es[j]) * bs[j, 0]
                hes[2 + 2 * i, 1 + 2 * j] = hes[1 + 2 * j, 2 + 2 * i] = \
                    self._n2 * np.sum(xes[i] * es[j]) * bs[i, 0]

        return mse, jac, hes


class ErrorWithFixedParameter():
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






#

