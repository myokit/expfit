#
# Single and multi-expontial fits
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import numpy as np

import expfit


def _decaying(p):
    """ Constraint for fitting decaying exponentials. """

    c = p[2::2]
    return np.all(c < 0) and np.all(c[1:] > c[:-1])


def fit1(t, v, plot=False):
    """
    Fits an exponential ``a + b * exp(c * t)`` to the time series ``(t, v)``,
    returning ``(a, b, c)``

    Example::

        t = np.linspace(0, 1, 100)
        v = 3 - 2 * np.exp(4 * t) + np.random.normal(0, 1, size=len(t))
        a, b, c = expfit.fit_single(t, v)
        print(a, b, c)

    """
    t, v = expfit.vet_series(t, v)

    # Transform to unit square, to avoid overflows etc
    tr = expfit.UnitSquareTransform(t, v)
    x, y = tr.x, tr.y

    # Create initial plot
    known = False
    try:
        if len(plot) == 3:  # pragma: no cover
            known = plot
            plot = True
    except TypeError:
        pass
    if plot:  # pragma: no cover
        import matplotlib.pyplot as plt
        if plot == 'simple':
            fig = plt.figure(figsize=(8, 4))
            ax0 = fig.add_subplot()
        else:
            fig = plt.figure(figsize=(9, 7.5))
            ax0 = fig.add_subplot(2, 1, 1)
        fig.subplots_adjust(0.095, 0.06, 0.995, 0.995, wspace=0.3, hspace=0.44)
        ax0.set_xlabel('x')
        ax0.set_ylabel('y')
        code, color = ('-', '#92cc92') if len(x) > 10 else ('x-', 'tab:green')
        ax0.plot(x, y, code, color=color, label='Transformed data')
    else:
        ax0 = None

    # Get an initial estimate
    at0, bt0, ct0 = expfit.estimate_initial_single(x, y, axes=ax0, vet=False)

    # Fit
    e = expfit.SingleExponentialError(x, y)
    with np.errstate(all='ignore'):
        r = expfit.fmin(e, (at0, bt0, ct0))
        if plot:  # pragma: no cover
            print(r)
    at, bt, ct = r.x

    # Detransform obtained parameters
    a, b, c = tr.detransform(at, bt, ct)

    if plot:  # pragma: no cover
        a0, b0, c0 = tr.detransform(at0, bt0, ct0)

        fit_param = f'{a:.3}, {b:.3}, {c:.3}'
        fit_label = f'rmse {np.sqrt(r.error):.4}'
        if r.success:
            fit_label = f'Fit ({fit_param}, {r.iterations} iter, {fit_label})'
        else:
            fit_label = f'Fit ({fit_param}, {r.message}, {fit_label})'

        ax0.plot(x, at0 + bt0 * np.exp(ct0 * x), '-',
                 label=f'Initial ({a0:.3}, {b0:.3}, {c0:.3})')
        ax0.plot(x, at + bt * np.exp(ct * x), '--', label=fit_label)
        ax0.legend()

        if plot != 'simple':
            lines = [
                f'Transformed Init: {a0:+.5e} {b0:+.5e} {c0:+.5e}',
                f'             Fit:  {a:+.5e} {b:+.5e} {c:+.5e}',
                f'Real-world  Init: {at0:+.5e} {bt0:+.5e} {ct0:+.5e}',
                f'             Fit:  {at:+.5e} {bt:+.5e} {ct:+.5e}']
            ax0.text(0.75, -0.38, '\n'.join(lines), transform=ax0.transAxes,
                     ha='right', font='monospace')

            ax1 = fig.add_subplot(2, 2, 3)
            ax1.set_xlabel('x')
            ax1.set_ylabel('Residuals')
            ax1.plot(x, y - (at0 + bt0 * np.exp(ct0 * x)), label='Initial')
            ax1.plot(x, y - (at + bt * np.exp(ct * x)), label='Fit')
            ax1.legend()

            ax2 = fig.add_subplot(2, 2, 4)
            ax2.set_xlabel('t')
            ax2.set_ylabel('v')
            label = 'Untransformed data'
            with np.errstate(divide='ignore'):
                if known:
                    label = f'{label} (tau={-1 / known[2]:+.3f})'
                ax2.plot(t, v, code, color=color, label=label)
                ax2.plot(t, a0 + b0 * np.exp(c0 * t), '-',
                         label=f'Initial (c={c0:+.3f}, tau={-1 / c0:+.3f})')
                ax2.plot(t, a + b * np.exp(c * t), '--',
                         label=f'fFit (c={c:+.3f}, tau={-1 / c:+.3f})')
            ax2.legend()

    return a, b, c


def fitd2(t, v, plot=False, vet=True):
    """
    Fits a double-exponential ``y = a + b0 * exp(c0 * x) + b1 * exp(c1 * x)``,
    where ``b0`` and ``b1`` have the same sign, ``c0`` and ``c1`` are both
    negative, and ``c1 > c0``.
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

