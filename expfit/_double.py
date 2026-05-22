#
# Single expontial fits
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import numpy as np

import expfit


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

    # Estimate the dominant rate
    tr = expfit.UnitSquareTransform(t, v)
    q0 = expfit.estimate_initial_single(tr.x, tr.y, vet=False)
    a0, b0, c0 = tr.detransform(q0)
    del tr, q0

    # Catch nans etc.
    if c0 == 0:
        return a0, b0, 0, 0, 0

    # Catch non-decaying
    if c0 > 0:
        raise RuntimeError(
            'Initial estimate for c > 0, exponential not decaying')

    # Assume dominant rate found, next rate will have smaller magnitude
    # Start with 2 times smaller, but increase if the rates converge
    d0 = b0
    e0 = c0
    p0 = np.array((a0, b0, c0, d0, e0), dtype=float)
    for i in range(1, 6):
        p0[4] *= 0.5
        e = expfit.MultiExponentialError(t, v)
        with np.errstate(all='ignore'):
            r = expfit.fmin(e, p0, constraint=_decaying)
            if plot:  # pragma: no cover
                print(r)
        p = r.x
        if p[2] / p[4] - 1 > 1e-3 and r.success:
            break

    if plot:  # pragma: no cover
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(9, 7.5))
        fig.subplots_adjust(0.095, 0.06, 0.995, 0.995, wspace=0.3, hspace=0.35)
        grd = fig.add_gridspec(2, 2, height_ratios=(2, 1))

        ax0 = fig.add_subplot(grd[0, :])
        ax0.set_xlabel('t')
        ax0.set_ylabel('v')

        # Show data, prepare to show fit
        code = '-' if len(t) > 10 else 'x-'
        ax0.plot(t, v, code, color='tab:blue', label='Data')
        f = expfit.exp
        label = f'rmse {np.sqrt(r.error):.4}'
        if r.success:
            label = f'Fit ({r.iterations} iter, {label})'
        else:
            label = f'Fit ({r.message}, {label})'

        # Show parameters
        pstr = lambda p: ' '.join(f'{i:+.5e}' for i in p)  # noqa
        ax0.text(0.5, -0.21, f'Init: {pstr(p0)}\n Fit: {pstr(p)}',
                 transform=ax0.transAxes, ha='center', font='monospace')

        # Try showing known solution
        try:
            assert len(plot) == 5
        except (TypeError, AssertionError):
            pass
        else:
            ax0.plot(t, f(t, (plot[0], plot[1], plot[2])), color='tab:green',
                     label=f'Known 1st (tau={1 / plot[2]:+.2g})',)
            ax0.plot(t, f(t, (plot[0], plot[3], plot[4])), color='tab:red',
                     label=f'Known 2nd (tau={1 / plot[4]:+.2g})')

        # Show fit
        ax0.plot(t, f(t, p), lw=1, color='k', label=label)
        ax0.plot(t, f(t, (p[0], p[1], p[2])), lw=1, ls='--',
                 color='#1f701f', label=f'Fit 1st (tau={1 / p[2]:+.2g})')
        lo, hi = expfit.ci(t, v, p, 2, constraint=_decaying)
        ax0.fill_between(
            t, f(t, (lo[0], lo[1], lo[2])), f(t, (hi[0], hi[1], hi[2])),
            color='#1f701f', alpha=0.3)
        ax0.plot(t, f(t, (p[0], p[3], p[4])), lw=1, ls='--',
                 color='#961b1c', label=f'Fit 2nd (tau={1 / p[4]:+.2g})')
        lo, hi = expfit.ci(t, v, p, 4, constraint=_decaying)
        ax0.fill_between(
            t, f(t, (lo[0], lo[3], lo[4])), f(t, (hi[0], hi[3], hi[4])),
            color='#961b1c', alpha=0.3)
        ax0.legend(framealpha=1, ncol=2)

        # Show single exponential estimate
        ax1 = fig.add_subplot(grd[1, 0])
        ax1.set_xlabel('t')
        ax1.set_ylabel('v')
        ax1.plot(t, v, code, color='tab:blue', label='Data')
        ax1.plot(t, f(t, (a0, b0, c0)), 'k--', lw=1, label='Initial')
        ax1.legend()

        # Show final fit residuals
        r = v - f(t, p)
        ax2 = fig.add_subplot(grd[1, 1])
        ax2.set_xlabel('t')
        ax2.set_ylabel('Residuals')
        ax2.plot(t, r, label='Residuals fit')
        ax2.legend()

    return p


def fit_double_decaying_tau(t, v):
    """
    Fits a double decaying exponential (see :meth:`fit_double_decaying`),
    returning two time constants along with confidence intervals.

    Returns ``(c1, c2, ci1, ci2)`` where ``cix`` is a tuple with the interval.
    """
    p = fit_double_decaying(t, v)
    c1 = p[2]
    c2 = p[4]

    if c1 == 0 or c2 == 0:
        # Instead of checking sign of zero and returning + or - inf, let numpy
        # handle it (but silently)
        with np.errstate(divide='ignore'):
            return -1 / c1, -1 / c2, (np.nan, np.nan), (np.nan, np.nan)

    lo, hi = expfit.ci(t, v, p, 2, constraint=_decaying)
    ci1 = np.sort(np.array((lo[2], hi[2])))
    lo, hi = expfit.ci(t, v, p, 4, constraint=_decaying)
    ci2 = np.sort(np.array((lo[4], hi[4])))
    return c1, c2, ci1, ci2

