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
        r = expfit.lm(f, (1, 1))
    r = expfit.ExponentialFit(x, y, r.x, f)
    return r


class TestCI(unittest.TestCase):
    """ Tests the confidence interval methods on a linear error """
    @classmethod
    def setUpClass(cls):
        # Create in each test and seed!
        cls.r = None

    def t(self, a0, b0, s0, n=51, delta=1e-2, ca=None, cb=None, plot=False):

        x = np.linspace(0, 10, n)
        y = a0 + b0 * x + self.r.normal(0, s0, size=x.shape)
        a, b = p = fit(x, y)
        cipa = p.ci_profile(0)  # Two solutions
        cipb = p.ci_profile(1)
        cifa = p.ci_fisher(0)  # One plus-minus term
        cifb = p.ci_fisher(1)

        if plot:  # pragma: no cover
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(11, 7.5))
            fig.subplots_adjust(0.075, 0.06, 0.99, 0.99, hspace=0.15)
            grid = fig.add_gridspec(2, 2, height_ratios=(3, 1))

            ax0 = fig.add_subplot(grid[0, :])
            ax0.plot(x, y, 'ko', label=f'Data (n={len(x)})')
            ax0.plot(x, a + b * x, label=f'Fit (a={a:.5g} b={b:.5g})')
            ax0.plot(x, a0 + b0 * x, label=f'True (a={a0}, b={b0})')

            # Show uncertainty in a
            alo1, ahi1 = cipa[0][0], cipa[1][0]
            alo2, ahi2 = p[0] - cifa, p[0] + cifa
            alo, ahi = min(alo1, alo2), max(ahi1, ahi2)
            xx, yy = p.profile(0, alo, ahi, 25)

            ax1 = fig.add_subplot(grid[1, 0])
            ax1.set_xlabel('a')
            ax1.set_ylabel('MSE')
            ax1.plot(xx, yy, color='tab:green', label='Profile')
            ax1.axvline(a0, color='k', label='True')
            ax1.axvline(alo1, color='tab:olive', lw=3,
                        label=f'PL CI ({alo1:.5g}, {ahi1:.5g})')
            ax1.axvline(ahi1, color='tab:olive', lw=3,)
            ax1.axvline(alo2, color='tab:pink', lw=2, ls='--',
                        label=f'FIM CI ({alo2:.5g}, {ahi2:.5g})')
            ax1.axvline(ahi2, color='tab:pink', lw=2, ls='--')

            # Forward predictions
            y0, y1 = cipa[0][0] + cipa[0][1] * x, cipa[1][0] + cipa[1][1] * x
            ax0.plot(x, y0, lw=1, color='tab:green')
            ax0.plot(x, y1, lw=1, color='tab:green', label='a predictions')
            ax0.fill_between(x, y0, y1, color='tab:green', alpha=0.1)

            # Add MSE for scanned area, without varying other parameters
            e = Linear1d(x, y)
            m = 100
            t = np.linspace(alo, ahi, 100)
            T = np.repeat(np.array(p).reshape((1, len(p))), m, axis=0)
            T[:, 0] = t
            V = [e(t)[0] for t in T]
            ax1.plot(t, V, label='MSE')

            # Add quadratic approximation around p
            mse, _, hes = e(p)
            T = np.zeros((len(p), m))
            T[0, :] = t - p[0]
            V = mse + 0.5 * np.array([i.dot(hes).dot(i.T) for i in T.T])
            ax1.plot(t, V, '--', label='Quadratic')

            # Repeat for b
            blo1, bhi1 = cipb[0][1], cipb[1][1]
            blo2, bhi2 = p[1] - cifb, p[1] + cifb
            blo, bhi = min(blo1, blo2), max(bhi1, bhi2)
            xx, yy = p.profile(1, blo, bhi, 25)

            ax2 = fig.add_subplot(grid[1, 1])
            ax2.set_xlabel('b')
            ax2.set_ylabel('MSE')
            ax2.plot(xx, yy, color='tab:purple', label='Profile')
            ax2.axvline(b0, color='k', label='True')
            ax2.axvline(blo1, color='tab:olive', lw=1,
                        label=f'PL CI ({alo1:.5g}, {ahi1:.5g})')
            ax2.axvline(bhi1, color='tab:olive', lw=1)
            ax2.axvline(blo2, color='tab:pink', lw=3, ls='--',
                        label=f'FIM CI ({alo2:.5g}, {ahi2:.5g})')
            ax2.axvline(bhi2, color='tab:pink', lw=3, ls='--')

            # Forward predictions
            y0, y1 = cipb[0][0] + cipb[0][1] * x, cipb[1][0] + cipb[1][1] * x
            ax0.plot(x, y0, lw=1, color='tab:purple')
            ax0.plot(x, y1, lw=1, color='tab:purple', label='b predictions')
            ax0.fill_between(x, y0, y1, color='tab:purple', alpha=0.1)

            # Add MSE for scanned area, without varying other parameters
            m = 100
            t = np.linspace(blo, bhi, 100)
            T = np.repeat(np.array(p).reshape((1, len(p))), m, axis=0)
            T[:, 1] = t
            V = [e(t)[0] for t in T]
            ax2.plot(t, V, label='MSE')

            # Add quadratic approximation around p
            mse, _, hes = e(p)
            T = np.zeros((len(p), m))
            T[1, :] = t - p[1]
            V = mse + 0.5 * np.array([i.dot(hes).dot(i.T) for i in T.T])
            ax2.plot(t, V, '--', label='Quadratic')

            # Finalise
            ax0.legend()
            ax1.legend(loc='upper center', framealpha=1)
            ax2.legend(loc='upper center', framealpha=1)
            plt.show()

            '''
            n = len(x)
            r = y - a - b * x
            sb = np.sqrt(np.sum(r**2) / (n - 2) / np.sum((x - np.mean(x))**2))
            sb = np.sqrt(np.sum(r**2) / (n - 2) / np.sum((x - np.mean(x))**2))
            sa = np.sqrt(np.sum(x**2) / n) * sb
            import scipy
            t90 = scipy.stats.t.ppf(0.95, n - 2)
            print(p.ci_fisher(0) / (t90 * sa))
            print(p.ci_fisher(1) / (t90 * sb))
            '''

        with self.subTest(a=a0, b=b0, s=s0, n=n):
            self.assertAlmostEqual(cipa[0][0], a - cifa, delta=delta)
            self.assertAlmostEqual(cipa[1][0], a + cifa, delta=delta)
            self.assertAlmostEqual(cipb[0][1], b - cifb, delta=delta)
            self.assertAlmostEqual(cipb[1][1], b + cifb, delta=delta)
            self.assertLess(cipa[0][0], a0)
            self.assertGreater(cipa[1][0], a0)
            self.assertLess(cipb[0][1], b0)
            self.assertGreater(cipb[1][1], b0)
            if ca is not None:
                self.assertAlmostEqual(cipa[0][0], ca[0], delta=delta)
                self.assertAlmostEqual(cipa[1][0], ca[1], delta=delta)
            if cb is not None:
                self.assertAlmostEqual(cipb[0][1], cb[0], delta=delta)
                self.assertAlmostEqual(cipb[1][1], cb[1], delta=delta)

    def test_linear(self):
        """ Test on linear problem """
        t = self.t
        self.r = np.random.default_rng(1)
        plot = False

        t(5, 3, 5, ca=(3.3591, 7.3179), cb=(2.5621, 3.2444), plot=plot)
        t(-70, -80, 50, plot=plot)

    def test_profile(self):
        """ Test .profile() """
        self.r = np.random.default_rng(1)
        plot = False

        a0, b0, s0 = -5, 10, 20
        x = np.linspace(0, 10, 111)
        y = a0 + b0 * x + self.r.normal(0, s0, size=x.shape)
        a, b = p = fit(x, y)

        cif = p.ci_fisher(0)
        alo, ahi = a - cif, a + cif
        xx, yy = p.profile(0, alo, ahi, evals=5)

        if plot:  # pragma: no cover
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(2, 1, 1)
            ax.plot(x, y, 'o')
            ax.plot(x, a + b * x)
            ax = fig.add_subplot(2, 1, 2)
            ax.plot(xx, yy)
            plt.show()

        self.assertEqual(list(xx), [
            alo,
            -8.164001404337903,
            -5.489527397607352,
            -2.815053390876802,
            ahi
        ])
        self.assertEqual(list(yy), [
            304.63807345375034,
            299.2006423113714,
            297.388165267725,
            299.200642322811,
            304.6380734766293,
        ])

    def test_double(self):
        # Test on double
        plot = False

        self.r = np.random.default_rng(1)
        p0 = np.array([1, -9, -2, -4, -7])
        n = 100
        t = np.linspace(0, 2, n)
        v = expfit.exp(t, p0)
        v += self.r.normal(0, 0.01 * abs(v[0] - v[-1]), size=n)

        p = expfit.fitd2(t, v, plot=p0 if plot else False)
        if plot:  # pragma: no cover
            import matplotlib.pyplot as plt
            plt.show()

        c1 = p.ci_fisher(2)
        self.assertAlmostEqual(c1, 0.173748, delta=1e-5)
        self.assertLess(p[2] - c1, p0[2])
        self.assertGreater(p[2] + c1, p0[2])

        c2 = p.ci_fisher(4)
        self.assertAlmostEqual(c2, 1.5617967, delta=1e-5)
        self.assertLess(p[4] - c2, p0[4])
        self.assertGreater(p[4] + c2, p0[4])

        c1 = p.ci_profile(2)
        self.assertAlmostEqual(c1[0][2], -2.1795243, delta=1e-5)
        self.assertAlmostEqual(c1[1][2], -1.8230414, delta=1e-5)
        self.assertLess(c1[0][2], p0[2])
        self.assertGreater(c1[1][2], p0[2])

        c2 = p.ci_profile(4)
        self.assertAlmostEqual(c2[0][4], -8.9935671, delta=1e-5)
        self.assertAlmostEqual(c2[1][4], -5.8337929, delta=1e-5)
        self.assertLess(c2[0][4], p0[4])
        self.assertGreater(c2[1][4], p0[4])

    def test_no_error(self):
        # Exponential fit without error
        e = expfit.ExponentialFit([1, 2], [3, 4], [5])
        self.assertFalse(e.ci_available())
        self.assertRaises(expfit.CIUnavailableError, e.ci_fisher, 0)
        self.assertRaises(expfit.CIUnavailableError, e.cov)
        self.assertRaises(expfit.CIUnavailableError, e.ci_profile, 0)
        self.assertRaises(expfit.CIUnavailableError, e.error)
        self.assertRaises(expfit.CIUnavailableError, e.profile, 0, -1, 1)
        self.assertRaises(expfit.CIUnavailableError, e.mse_cutoff, 0)

        # Test with error returning function
        x, y = np.array([1, 2]), np.array([3, 4])
        f = Linear1d(x, y)
        e = expfit.ExponentialFit(x, y, [5], f)
        self.assertTrue(e.ci_available())
        self.assertIs(e.error(), f)

    def test_mse_cutoff(self):
        # MSE cut-off for profile CI, with 90% chi squared stat

        x = np.arange(0, 10, 1)
        y = 3 * x
        f = lambda p: (11, [11], [[11]])
        e = expfit.ExponentialFit(x, y, [7], f)
        self.assertEqual(e.mse_cutoff(), (1 + 2.706 / 10) * 11)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
