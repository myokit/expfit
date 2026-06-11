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
        """ Calculate only the MSE (for a single point). """
        return self._m * np.sum(
            (p[0] - self._y + p[1] * np.exp(p[2] * self._x))**2)


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
        b = p[1::2].reshape((m, 1))        # (m, 1)
        c = p[2::2].reshape((m, 1))        # (m, 1)

        # MSE
        e = np.exp(c * self._x)               # (m, n)  e^(cx)
        be = b * e                            # (m, n) be^(cx)
        f = a - self._y + np.sum(be, axis=0)  # (n, ) a - y + sum_j(be^(cx))
        mse = np.sum(f**2) * self._ni

        # Jacobian
        ex = e * self._x
        jac = np.zeros(d)
        jac[0] = self._n2 * np.sum(f)
        jac[1::2] = self._n2 * np.sum(f * e, axis=1)
        jac[2::2] = self._n2 * np.sum(f * ex, axis=1) * b.T

        # Hessian
        hes = np.zeros((d, d))

        # aa, ab, ac
        hes[0, 0] = 2
        hes[0, 1::2] = hes[1::2, 0] = self._n2 * np.sum(e, axis=1)
        hes[0, 2::2] = hes[2::2, 0] = self._n2 * np.sum(ex, axis=1) * b.T
        for i in range(m):
            fbex = (f + be[i]) * ex[i]
            # bi^2, ci^2, and bi*ci
            hes[1 + 2 * i, 1 + 2 * i] = self._n2 * np.sum(e[i]**2)
            hes[2 + 2 * i, 2 + 2 * i] = \
                self._n2 * np.sum(fbex * self._x) * b[i, 0]
            hes[1 + 2 * i, 2 + 2 * i] = hes[2 + 2 * i, 1 + 2 * i] = \
                self._n2 * np.sum(fbex)

            for j in range(i + 1, m):
                exe = np.sum(ex[i] * e[j])
                # bi*bj, ci*cj, bi*cj, bj*ci
                hes[1 + 2 * i, 1 + 2 * j] = hes[1 + 2 * j, 1 + 2 * i] = \
                    self._n2 * np.sum(e[i] * e[j])
                hes[2 + 2 * i, 2 + 2 * j] = hes[2 + 2 * j, 2 + 2 * i] = \
                    self._n2 * np.sum(ex[i] * ex[j]) * b[i, 0] * b[j, 0]
                hes[1 + 2 * i, 2 + 2 * j] = hes[2 + 2 * j, 1 + 2 * i] = \
                    self._n2 * exe * b[j, 0]
                hes[2 + 2 * i, 1 + 2 * j] = hes[1 + 2 * j, 2 + 2 * i] = \
                    self._n2 * exe * b[i, 0]

        return mse, jac, hes

    def mse(self, p):
        """ Calculate only the MSE (for a single point). """
        # Unpack
        d = len(p)
        assert (d - 1) % 2 == 0 and d > 1
        m = (d - 1) // 2

        p = np.asarray(p)
        a = p[0]
        bs = p[1::2].reshape((m, 1))        # (m, 1)
        cs = p[2::2].reshape((m, 1))        # (m, 1)

        # MSE
        return self._ni * np.sum((
            a - self._y + np.sum(bs * np.exp(np.outer(cs, self._x)), axis=0)
        )**2)


class DecayingMultiExponentialError():
    """
    Callable class returning the MSE and its Jacobian and Hessian for a
    multi-exponential ``y = a + b_i * exp(-exp(q_i) * x)`` fit with parameters
    ``p = (a, b_1, q_1, b_2, q_2, ...)``.
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
        qs = p[2::2].reshape((m, 1))        # (m, 1)

        # MSE
        n = len(self._x)
        ninv2 = 2 / n
        eqs = np.exp(qs)
        es = np.exp(-eqs * self._x)
        bes = bs * es
        fs = a - self._y + np.sum(bes, axis=0)
        mse = np.sum(fs**2) / n

        # Jacobian
        jac = np.zeros(d)
        xes = es * self._x * eqs
        jac[0] = ninv2 * np.sum(fs)
        jac[1::2] = ninv2 * np.sum(fs * es, axis=1)
        jac[2::2] = -ninv2 * np.sum(fs * xes, axis=1) * bs.T

        # Hessian
        hes = np.zeros((d, d))
        # aa, ab, ac
        hes[0, 0] = 2
        hes[0, 1::2] = hes[1::2, 0] = ninv2 * np.sum(es, axis=1)
        hes[0, 2::2] = hes[2::2, 0] = -ninv2 * np.sum(xes, axis=1) * bs.T
        for i in range(m):
            # bi^2, ci^2, and bi*ci
            hes[1 + 2 * i, 1 + 2 * i] = ninv2 * np.sum(es[i]**2)
            hes[2 + 2 * i, 2 + 2 * i] = ninv2 * bs[i, 0] * np.sum(
                ((fs + bes[i]) * self._x * eqs[i] - fs) * xes[i])
            hes[1 + 2 * i, 2 + 2 * i] = hes[2 + 2 * i, 1 + 2 * i] = \
                -ninv2 * np.sum((fs + bes[i]) * xes[i])
            for j in range(i + 1, m):
                # bi*bj, ci*cj, bi*cj, bj*ci
                hes[1 + 2 * i, 1 + 2 * j] = hes[1 + 2 * j, 1 + 2 * i] = \
                    ninv2 * np.sum(es[i] * es[j])
                hes[2 + 2 * i, 2 + 2 * j] = hes[2 + 2 * j, 2 + 2 * i] = \
                    ninv2 * np.sum(xes[i] * xes[j]) * bs[i, 0] * bs[j, 0]
                hes[1 + 2 * i, 2 + 2 * j] = hes[2 + 2 * j, 1 + 2 * i] = \
                    -ninv2 * np.sum(xes[j] * es[i]) * bs[j, 0]
                hes[2 + 2 * i, 1 + 2 * j] = hes[1 + 2 * j, 2 + 2 * i] = \
                    -ninv2 * np.sum(xes[i] * es[j]) * bs[i, 0]

        return mse, jac, hes


class ErrorWithFixedParameter():
    """
    Wraps around an error class and fixes one parameter.

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


class DecayingEqualSignConstraint():
    """
    Constraint for fitting decaying exponentials ``a + b_i * exp(c_i * x)``,
    where all ``b`` have the same sign.

    In full:

    1. All ``b`` have the same sign.
    2. All ``c`` are negative.
    3. The dominant (slowest) exponential has the smallest absolute ``c`` (or
       largest tau). Exponentials are ordered by dominance, so that
       ``abs(c[i]) < abs(c[i + 1])`` or ``c[i] > c[i + 1]``

    """
    def __call__(self, p):
        b, c = p[1::2], p[2::2]
        return (np.all(c < 0) and np.all(c[:-1] > c[1:]) and
                (np.all(b <= 0) or np.all(b >= 0)))


class EqualSignConstraint():
    """
    Constraint for fitting decaying exponentials


     ``a + b_i * exp(c_i * x)``,
    where all ``b`` have the same sign.

    In full:

    1. All ``b`` have the same sign.
    2. All ``c`` are negative.
    3. The dominant (slowest) exponential has the smallest absolute ``c`` (or
       largest tau). Exponentials are ordered by dominance, so that
       ``abs(c[i]) < abs(c[i + 1])`` or ``c[i] > c[i + 1]``

    """
    def __call__(self, p):
        b = p[1::2]
        return np.all(b <= 0) or np.all(b >= 0)


class DecayingOppositeSignConstraint():
    """
    Constraint for fitting decaying exponentials ``a + b_i * exp(c_i * x)``
    where all ``b_i`` have the same sign until ``i = n_initial``, after which
    they all have the opposite sign.

    In full:





    1. All ``b`` have the same sign.
    2. All ``c`` are negative.
    3. The dominant (slowest) exponential has the smallest absolute ``c`` (or
       largest tau). Exponentials are ordered by dominance, so that
       ``abs(c[i]) < abs(c[i + 1])`` or ``c[i] > c[i + 1]``

    "d11" exponentials: ``b[0] * b[1] < 0``,
    ``c[0] < 0``, ``c[1] < 0``, and ``c[1] > c[0]``,
    """
    #def __init__(self, i):

    def __call__(self, p):
        #return p[2] < 0 and p[4] < 0 and p[4] > p[2] and p[1] * p[3] < 0
        return p[2] < 0 and p[4] < 0 and p[1] * p[3] < 0


class D11Constraint():
    """
    Constraint for fitting "d11" exponentials: ``b[0] * b[1] < 0``,
    ``c[0] < 0``, ``c[1] < 0``, and ``c[1] > c[0]``,
    """
    def __call__(self, p):
        return p[2] < 0 and p[4] < 0 and p[4] > p[2] and p[1] * p[3] < 0


class ConstraintWithFixedParameter():
    """
    Wraps around a constraint and fixes one parameter.

    Arguments:

    ``error``
        The error to wrap.
    ``p``
        The best solution
    ``ifix``
        The index in the parameter vector of the parameter to hold fixed.

    """
    def __init__(self, constraint, p, ifix):
        self._c = constraint
        self._p = np.copy(p)
        self._i = int(ifix)

    def __call__(self, p):
        self._p[:self._i] = p[:self._i]
        self._p[self._i + 1:] = p[self._i:]
        return self._c(self._p)

