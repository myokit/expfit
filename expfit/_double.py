#
# Single expontial fits
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import numpy as np

import expfit


def _rmse_double_decaying(x, y, p):
    rb = p[1] / p[3]    # >0 to keep sign equal
    rc = p[2] / p[4]    # >1 to keep sign equal, and c > e
    if rb < 0 or rc < 1 or p[2] > 0:
        return np.inf  # TODO
    return np.sqrt(np.sum(
        (y - p[0] - p[1] * np.exp(p[2] * x) - p[3] * np.exp(p[4] * x))**2
    ) / len(x))


def _decaying(p):
    c = p[2::2]
    return np.all(c < 0) and np.all(c[1:] > c[:-1])


def fit_double_decaying(t, v, plot=False, vet=True):
    """
    Fits a double-exponential ``y = a + b * exp(c * x) + d * exp(e * x)``,
    where ``b`` and ``d`` have the same sign, and ``c`` and ``e`` are both
    negative.
    """
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
        fig = plt.figure(figsize=(9, 7.5))
        fig.subplots_adjust(0.095, 0.06, 0.995, 0.995, wspace=0.3, hspace=0.2)
        grd = fig.add_gridspec(2, 2, height_ratios=(2, 1))

        ax0 = fig.add_subplot(grd[0, :])
        ax0.set_xlabel('x')
        ax0.set_ylabel('y')
        code, color = ('-', '#92cc92') if len(x) > 10 else ('x-', 'tab:green')
        ax0.plot(x, y, code, color=color, label='Transformed data')

    # Get an initial estimate, for the dominant rate
    at0, bt0, ct0 = expfit.estimate_initial_single(x, y, vet=False)

    # Catch nans etc.
    if ct0 == 0:
        return v[0] + at0 * rv, bt0 * rv, 0., 0., 0.

    # Assume dominant rate found, next rate will have smaller magnitude, but
    # bigger multiplier to stay visible
    # Start with 2 times smaller, but increase if the rates converge
    dt0 = bt0 * 0.1
    bt0 *= 0.9
    for i in range(-1, -6, -1):
        et0 = ct0 * 2**i
        p0 = np.array((at0, bt0, ct0, dt0, et0), dtype=float)

        e = expfit.MultiExponentialError(x, y)
        with np.errstate(all='ignore'):
            r = expfit.fmin(e, p0, constraint=_decaying)
            if plot:  # pragma: no cover
                print(r)
        at, bt, ct, dt, et = r.x
        if ct / et - 1 > 1e-3 and r.success:
            break

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

        try:
            known = False
            if len(plot) == 5:
                ak, bk, ck, dk, ek = plot
                plot = known = True
                akt = (ak - v[0]) / rv
                bkt = bk / rv * np.exp(ck * t[0])
                ckt = ck * rt
                dkt = dk / rv * np.exp(ek * t[0])
                ekt = ek * rt
        except TypeError:
            pass

        print()
        print(f'Init: {a0:+.5e} {b0:+.5e} {c0:+.5e} {d0:+.5e} {e0:+.5e}')
        print(f'Fit:  {a:+.5e} {b:+.5e} {c:+.5e} {d:+.5e} {e:+.5e}')

        ax0.plot(x, at0 + bt0 * np.exp(ct0 * x), '-', label='Initial')
        if known:
            ax0.plot(x, akt * np.ones(x.shape), 'k--', label='Known offset',
                     zorder=10)
            ax0.plot(
                x, bkt * np.exp(ckt * x), 'k-.', label='Known 1st', zorder=10)
            ax0.plot(
                x, dkt * np.exp(ekt * x), 'k:', label='Known 2nd', zorder=10)
        ax0.plot([x[0], x[-1]], [at, at], '-', label='Fit offset')
        ax0.plot(x, bt * np.exp(ct * x), '-', label='Fit 1st')
        ax0.plot(x, dt * np.exp(et * x), '-', label='Fit 2nd')
        ax0.plot(x, at + bt * np.exp(ct * x) + dt * np.exp(et * x), '-',
                 label='Fit')
        ax0.legend(framealpha=1, ncol=2)

        r = y - (at + bt * np.exp(ct * x) + dt * np.exp(et * x))
        ax2 = fig.add_subplot(grd[1, 0])
        ax2.set_xlabel('x')
        ax2.set_ylabel('Residuals')
        ax2.plot(x, r, label='Residuals fit')
        ax2.legend()

        ax3 = fig.add_subplot(grd[1, 1])
        ax3.set_xlabel('t')
        ax3.set_ylabel('v')
        label = 'Untransformed'
        if known:
            label = f'{label} (tau1={-1 / ck:.3g}, tau2={-1 / ek:.3g})'
        ax3.plot(t, v, code, color=color, label=label)
        ax3.plot(t, a0 + b0 * np.exp(c0 * t), '-',
                 label=f'Initial (tau={-1 / c0:.3g})')
        ax3.plot(t, a + b * np.exp(c * t) + d * np.exp(e * t), '--',
                 label=f'Fit (tau1={-1 / c:.3g}, tau2={-1 / e:.3g})')
        ax3.legend()

    return a, b, c, d, e

