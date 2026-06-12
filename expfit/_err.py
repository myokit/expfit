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
            fbeex = (f + be[i]) * ex[i]
            # bi^2, ci^2, and bi*ci
            hes[1 + 2 * i, 1 + 2 * i] = self._n2 * np.sum(e[i]**2)
            hes[2 + 2 * i, 2 + 2 * i] = \
                self._n2 * np.sum(fbeex * self._x) * b[i, 0]
            hes[1 + 2 * i, 2 + 2 * i] = hes[2 + 2 * i, 1 + 2 * i] = \
                self._n2 * np.sum(fbeex)

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

    TODO
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
        b = p[1::2].reshape((m, 1))           # (m, 1)
        c = -np.exp(p[2::2]).reshape((m, 1))  # (m, 1)

        # MSE
        e = np.exp(c * self._x)               # (m, n)
        be = b * e                            # (m, n)
        f = a - self._y + np.sum(be, axis=0)  # (n, )
        mse = np.sum(f**2) * self._ni

        # Jacobian
        ex = e * self._x  # (m, n)
        bcT = (b * c).T   # (1, m)
        jac = np.zeros(d)
        jac[0] = self._n2 * np.sum(f)
        jac[1::2] = self._n2 * np.sum(f * e, axis=1)
        jac[2::2] = self._n2 * np.sum(f * ex, axis=1) * bcT

        # Hessian
        fbex = (f + be) * self._x  # (m, n)
        # aa, ab, ac
        hes = np.zeros((d, d))
        hes[0, 0] = 2
        hes[0, 1::2] = hes[1::2, 0] = self._n2 * np.sum(e, axis=1)
        hes[0, 2::2] = hes[2::2, 0] = self._n2 * np.sum(ex, axis=1) * bcT
        for i in range(m):
            # bi^2, ci^2, sand bi*ci
            hes[1 + 2 * i, 1 + 2 * i] = self._n2 * np.sum(e[i]**2)
            hes[2 + 2 * i, 2 + 2 * i] = \
                self._n2 * np.sum((fbex[i] * c[i, 0] + f) * ex[i]) * bcT[0, i]
            hes[1 + 2 * i, 2 + 2 * i] = hes[2 + 2 * i, 1 + 2 * i] = \
                self._n2 * np.sum(fbex[i] * e[i]) * c[i, 0]
            for j in range(i + 1, m):
                eij = e[i] * e[j]
                eijx = eij * self._x
                # bi*bj, ci*cj, bi*cj, bj*ci
                hes[1 + 2 * i, 1 + 2 * j] = hes[1 + 2 * j, 1 + 2 * i] = \
                    self._n2 * np.sum(eij)
                hes[2 + 2 * i, 2 + 2 * j] = hes[2 + 2 * j, 2 + 2 * i] = \
                    self._n2 * np.sum(eijx * self._x) * bcT[0, i] * bcT[0, j]
                hes[1 + 2 * i, 2 + 2 * j] = hes[2 + 2 * j, 1 + 2 * i] = \
                    self._n2 * np.sum(eijx) * bcT[0, j]
                hes[2 + 2 * i, 1 + 2 * j] = hes[1 + 2 * j, 2 + 2 * i] = \
                    self._n2 * np.sum(eijx) * bcT[0, i]

        return mse, jac, hes


class SignedDecayingMultiExponentialError():
    """
    Callable class returning the MSE and its Jacobian and Hessian for a
    multi-exponential ``y = a + b_i * exp(-exp(q_i) * x)`` fit with parameters
    ``p = (a, b_1, q_1, b_2, q_2, ...)``.

    TODO
    """
    def __init__(self, x, y, z):
        self._x = x
        self._y = y
        self._ni = 1 / len(x)
        self._n2 = 2 * self._ni

        self._m = self._np = None
        try:
            self._m = len(z)
        except AttributeError:
            self._z = float(z)
            if not (z == 1 or z == -1):
                raise ValueError('z can only be 1 or -1')
        else:
            self._z = np.array(z)
            if len(self._z.shape) > 1:
                raise ValueError('z must be a scalar or a 1-d array')
            if not np.all(np.logical_or(self._z == 1, self._z == -1)):
                raise ValueError('All entries in z must be 1 or -1')
            self._np = 1 + 2 * self._m

    def __call__(self, p):
        if self._m is None:
            d = len(p)
            assert (d - 1) % 2 == 0 and d > 1
            m = (d - 1) // 2
        else:
            d = self._np
            m = self._m
            assert len(p) == d

        # Unpack
        p = np.asarray(p)
        a = p[0]
        b = (self._z * np.exp(p[1::2])).reshape((m, 1))  # (m, 1)
        c = -np.exp(p[2::2]).reshape((m, 1))             # (m, 1)

        # MSE
        e = np.exp(c * self._x)               # (m, n)
        be = b * e                            # (m, n)
        f = a - self._y + np.sum(be, axis=0)  # (n, )
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


'''
class SignedDecayingMultiExponentialErrorB():
    """
    Callable class returning the MSE and its Jacobian and Hessian for a
    multi-exponential ``y = a + b_i * exp(-exp(q_i) * x)`` fit with parameters
    ``p = (a, b_1, q_1, b_2, q_2, ...)``.

    TODO
    """
    def __init__(self, x, y, z, p0):
        self._x = x
        self._y = y
        self._ni = 1 / len(x)
        self._n2 = 2 * self._ni

        p0 = np.array(p0)
        d = len(p0)
        assert (d - 1) % 2 == 0 and d > 1
        m = (d - 1) // 2
        self._a = p0[0]
        self._c = -np.exp(p0[2::2]).reshape((m, 1))             # (m, 1)
        self._m = m

        # TODO
        self._z = z

    def __call__(self, p):
        if self._m is None:
            d = m = len(p)
        else:
            d = m = self._m

        # Unpack
        p = np.asarray(p)
        b = (self._z * np.exp(p)).reshape((m, 1))  # (m, 1)

        # MSE
        e = np.exp(self._c * self._x)         # (m, n)
        be = b * e                            # (m, n)
        f = self._a - self._y + np.sum(be, axis=0)  # (n, )
        mse = np.sum(f**2) * self._ni

        # Jacobian
        ex = e * self._x        # (m, n)
        bcT = (b * self._c).T   # (1, m)
        jac = self._n2 * np.sum(f * e, axis=1) * b.T
        jac = jac[0]

        # Hessian
        fbe = (f + be)        # (m, n)
        fbex = fbe * self._x  # (m, n)
        # aa, ab, ac
        hes = np.zeros((d, d))
        for i in range(m):
            # bi^2
            hes[i, i] = \
                self._n2 * np.sum(fbe[i] * e[i]) * b[i, 0]
            for j in range(i + 1, m):
                # bi*bj
                hes[i, j] = hes[j, i] = \
                    self._n2 * np.sum(e[i] * e[j]) * b[i, 0] * b[j, 0]

        return mse, jac, hes
'''


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

