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


def _rmse_double_constrained_unidirectional(x, y, p):
    rb = p[1] / p[3]    # >0 to keep sign equal
    rc = p[2] / p[4]    # >1 to keep sign equal, and c > e
    if rb < 0 or rc < 1:
        return np.inf  # TODO
    return np.sqrt(np.sum((
        y - p[0] - p[1] * np.exp(p[2] * x) - p[3] * np.exp(p[4] * x))**2))


def fit_double(t, v, plot=False, vet=True):
    if vet:
        t, v = expfit.vet_series(t, v)

    # Transform to unit square, to avoid overflows and get useable numbers
    rt, rv = (t[-1] - t[0]), (v[-1] - v[0])
    if rv == 0:
        rv = 1
    x, y = (t - t[0]) / rt, (v - v[0]) / rv

    # Create initial plot
    if plot:  # pragma: no cover
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 9))
        fig.subplots_adjust(0.095, 0.06, 0.995, 0.995, wspace=0.3, hspace=0.2)
        ax0 = fig.add_subplot(3, 1, 1)
        ax0.set_xlabel('x')
        ax0.set_ylabel('y')
        code, color = ('-', '#92cc92') if len(x) > 10 else ('x-', 'tab:green')
        ax0.plot(x, y, code, color=color, label='Transformed data')
    else:
        ax0 = None

    # Get an initial estimate, for the dominant rate
    at0, bt0, ct0 = expfit.estimate_initial_single(x, y, axes=ax0, vet=False)

    # Attempt estimate over residuals
    #at1, dt0, et0 = expfit.estimate_initial_single(x, r, vet=False)

    # Assume dominant rate found, next rate will have smaller magnitude, but
    # bigger multiplier to stay visible
    dt0, et0 = bt0, ct0 / 2

    # Fit
    with np.errstate(all='ignore'):
        res = fmin(
            lambda p: _rmse_double_constrained_unidirectional(x, y, p),
            (at0, bt0, ct0, dt0, et0))
    at, bt, ct, dt, et = res.x

    # Detransform
    a = v[0] + at * rv
    b = bt * rv * np.exp(-ct * t[0] / rt)
    c = ct / rt
    d = dt * rv * np.exp(-et * t[0] / rt)
    e = et / rt

    if plot:  # pragma: no cover
        a0 = v[0] + at0 * rv
        b0 = bt0 * rv * np.exp(-ct0 * t[0] / rt)
        c0 = ct0 / rt
        d0 = dt0 * rv * np.exp(-et0 * t[0] / rt)
        e0 = et0 / rt

        print()
        print(f'Init: {a0:+.5e} {b0:+.5e} {c0:+.5e} {d0:+.5e} {e0:+.5e}')
        print(f'Fit:  {a:+.5e} {b:+.5e} {c:+.5e} {d:+.5e} {e:+.5e}')

        ax0.plot(x, at0 + bt0 * np.exp(ct0 * x), '-', label='Initial')
        ax0.plot([x[0], x[-1]], [at, at], '-', label='Fit 0th')
        ax0.plot(x, bt * np.exp(ct * x), '-', label='Fit 1st')
        ax0.plot(x, dt * np.exp(et * x), '-', label='Fit 2nd')
        ax0.plot(x, at + bt * np.exp(ct * x) + dt * np.exp(et * x), '-',
                 label='Fit')
        ax0.legend()

        r0 = y - (at0 + bt0 * np.exp(ct0 * x))
        r1 = y - (at + bt * np.exp(ct * x))
        r2 = r1 - (dt * np.exp(et * x))

        ax1 = fig.add_subplot(3, 1, 2)
        ax1.set_xlabel('x')
        ax1.set_ylabel('Residuals')
        ax1.plot(x, r0, label='Residuals initial')
        ax1.plot(x, r1, label='Residuals fit 1st')
        ax1.legend()

        ax2 = fig.add_subplot(3, 2, 5)
        ax2.set_xlabel('x')
        ax2.set_ylabel('Residuals')
        ax2.plot(x, r2, label='Residuals fit')
        ax2.legend()

        ax3 = fig.add_subplot(3, 2, 6)
        ax3.set_xlabel('t')
        ax3.set_ylabel('v')
        ax3.plot(t, v, code, color=color, label='Untransformed data')
        ax3.plot(t, a0 + b0 * np.exp(c0 * t), '-', label='Initial')
        ax3.plot(t, a + b * np.exp(c * t) + d * np.exp(e * t), '--',
                 label='Fit')
        ax3.legend()

    # Assume

    #TODO

    return 0, 0, 0, 0, 0

