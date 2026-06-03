#!/usr/bin/env python3
#
# Tests for the confidence interval methods
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import unittest

import numpy as np

import expfit


class Linear1d():
    """
    Callable class returning the MSE and its Jacobian and Hessian for a line
    through the origin ``y = a + b * x``.
    """
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._m = 1 / len(x)

        self._h = np.zeros((2, 2))
        self._h[0, 0] = 2
        self._h[1, 1] = 2 * self._m * np.sum(self._x**2)
        self._h[1, 0] = self._h[0, 1] = 2 * self._m * np.sum(self._x)

    def __call__(self, p):
        a, b = p

        r = a + b * self._x - self._y  # Sign matters for jac
        e = self._m * np.sum(r**2)
        jac = np.array([2 * self._m * np.sum(r),
                        2 * self._m * np.sum(r * self._x)])
        hes = np.copy(self._h)
        return e, jac, hes


def fit(x, y):
    """
    Fit the series ``(x, y)`` with a linear model, returning an object that can
    do CI.
    """
    x, y = expfit.vet_series(x, y)
    f = Linear1d(x, y)
    with np.errstate(all='ignore'):
        r = expfit.fmin(f, (1, 1))
    r = expfit.ExponentialFit(x, y, r.x)
    r._err_class = Linear1d
    return r


class TestCI(unittest.TestCase):
    """ Tests the confidence interval methods on a linear error """

    def test_ci(self):

        import matplotlib.pyplot as plt

        rng = np.random.default_rng(1)
        x = np.linspace(0, 10, 51)
        at, bt = 5, 3
        y = at + bt * x + rng.normal(0, 2, size=x.shape)
        a, b = p = fit(x, y)

        fig = plt.figure(figsize=(11, 7.5))
        fig.subplots_adjust(0.075, 0.06, 0.99, 0.99, hspace=0.15)
        grid = fig.add_gridspec(2, 2, height_ratios=(3, 1))

        ax0 = fig.add_subplot(grid[0, :])
        ax0.plot(x, y, 'o', label=f'Data (n={len(x)}, a={at}, b={bt})')
        ax0.plot(x, a + b * x, label=f'a + b x (a={a:.2g} b={b:.2g})')
        ax0.legend()

        cip = p.ci_profile(0)
        cif = p.ci_fisher(0)
        alo1, ahi1 = cip[0][0], cip[1][0]
        alo2, ahi2 = p[0] - cif, p[0] + cif
        alo, ahi = min(alo1, alo2), max(ahi1, ahi2)
        #xx, yy = p.profile(0, alo, ahi, 25)

        f = Linear1d(x, y)
        print(cip[0], f(cip[0])[0])
        print(cip[1], f(cip[1])[0])

        ax1 = fig.add_subplot(grid[1, 0])
        ax1.set_xlabel('a')
        ax1.set_ylabel('MSE')
        #ax1.plot(xx, yy)
        ax1.axvline(p[0], color='gray', label='True')
        ax1.axvline(alo1, color='k', lw=1,
                    label=f'PL ({alo1:.2g}, {ahi1:.2g})')
        ax1.axvline(ahi1, color='k', lw=1)
        ax1.axvline(alo2, color='k', lw=1, ls='--',
                    label=f'FIM ({alo2:.2g}, {ahi2:.2g})')
        ax1.axvline(ahi2, color='k', lw=1, ls='--')
        ax1.legend(loc='lower right', framealpha=1)

        '''
        cip = p.ci_profile(1)
        cif = p.ci_fisher(1)
        alo1, ahi1 = cip[0][1], cip[1][1]
        alo2, ahi2 = p[1] - cif, p[1] + cif
        alo, ahi = min(alo1, alo2), max(ahi1, ahi2)
        xx, yy = p.profile(1, alo, ahi, 25)

        ax2 = fig.add_subplot(grid[1, 1])
        ax2.set_xlabel('b')
        ax2.set_ylabel('MSE')
        ax2.plot(xx, yy)
        ax2.axvline(p[1], color='gray', label='True')
        ax2.axvline(alo1, color='k', lw=1,
                    label=f'PL ({alo1:.2g}, {ahi1:.2g})')
        ax2.axvline(ahi1, color='k', lw=1)
        ax2.axvline(alo2, color='k', lw=1, ls='--',
                    label=f'FIM ({alo2:.2g}, {ahi2:.2g})')
        ax2.axvline(ahi2, color='k', lw=1, ls='--')
        ax2.legend(loc='lower right', framealpha=1)
        '''

        plt.show()






if __name__ == '__main__':  # pragma: no cover
    unittest.main()
