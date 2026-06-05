#
# Single and multi-expontial fits
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import numpy as np

import expfit


C1 = 'tab:red'
D1 = '#961b1c'
C2 = 'tab:purple'
D2 = '#5b3383'
# '#1f701f'


def fit1(t, v, plot=False):
    """
    Fits an exponential ``a + b * exp(c * t)`` to the time series ``(t, v)``,
    returning ``(a, b, c)``

    Example::

        t = np.linspace(0, 1, 100)
        v = 3 - 2 * np.exp(4 * t) + np.random.normal(0, 1, size=len(t))
        a, b, c = expfit.fit_single(t, v)
        print(a, b, c)

    Arguments:

    ``t``, ``v``
        The time series
    ``plot``
        Optional parameter to create a plot of the method's workings. Can be a
        boolean or the string "simple" for a reduced plot.

    Returns an :class:`ExponentialFit`.
    """
    t, v = expfit.vet_series(t, v)

    # Transform to unit square, to avoid overflows etc
    tr = expfit.UnitSquareTransform(t, v)

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
        fig.subplots_adjust(0.11, 0.06, 0.995, 0.995, wspace=0.3, hspace=0.44)
        ax0.set_xlabel('x')
        ax0.set_ylabel('y')
        ls, color = ('-', '#92cc92') if len(tr.x) > 10 else ('x-', 'tab:green')
        ax0.plot(tr.x, tr.y, ls, color=color, label='Transformed data')
    else:
        ax0 = None

    # Get an initial estimate (in transformed space)
    at0, bt0, ct0 = expfit.estimate_initial_single(
        tr.x, tr.y, axes=ax0, vet=False)

    # Fit (in transformed space)
    e = expfit.SingleExponentialError(tr.x, tr.y)
    with np.errstate(all='ignore'):
        r = expfit.lm(e, (at0, bt0, ct0))
        if plot:  # pragma: no cover
            print(r)
    at, bt, ct = r.x

    # Detransform obtained parameters, create result object
    p = expfit.ExponentialFit(
        t, v, tr.detransform(at, bt, ct), expfit.SingleExponentialError(t, v))

    if plot:  # pragma: no cover
        p0 = expfit.ExponentialFit(t, v, tr.detransform(at0, bt0, ct0))
        q0 = expfit.ExponentialFit(tr.x, tr.y, (at0, bt0, ct0))
        q = expfit.ExponentialFit(tr.x, tr.y, (at, bt, ct))

        strest = ', '.join(f'{i:.3}' for i in q0)
        strq = ', '.join(f'{i:.3}' for i in q)
        stre = f'rmse {np.sqrt(r.error):.4}'
        if r.success:
            strfit = f'{strq}, {r.iterations} iter, {stre}'
        else:
            strfit = f'{strq}, {r.message}, {stre}'

        e = expfit.exp
        ax0.plot(tr.x, e(tr.x, q0), '-', label=f'Initial ({strest})')
        ax0.plot(tr.x, e(tr.x, q), '--', label=f'Fit ({strfit})')
        ax0.legend()

        if plot != 'simple':
            lines = [f'Transformed Init: {q0}', f'             Fit:  {q}',
                     f'Real-world  Init: {p0}', f'             Fit:  {p}']
            ax0.text(0.75, -0.38, '\n'.join(lines), transform=ax0.transAxes,
                     ha='right', font='monospace')

            ax1 = fig.add_subplot(2, 2, 3)
            ax1.set_xlabel('x')
            ax1.set_ylabel('Residuals (transformed)')
            ax1.plot(tr.x, tr.y - e(tr.x, q0), label='Initial')
            ax1.plot(tr.x, tr.y - e(tr.x, q), label='Fit')
            ax1.legend()

            ax2 = fig.add_subplot(2, 2, 4)
            ax2.set_xlabel('t')
            ax2.set_ylabel('v')
            label = 'Untransformed data'
            with np.errstate(divide='ignore'):
                if known:
                    label = f'{label} (tau={-1 / known[2]:+.3f})'
                ax2.plot(t, v, ls, color=color, label=label)
                strc0 = f'c={p0[2]:+.3f}, tau={-1 / p0[2]:+.3f}'
                strc = f'c={p[2]:+.3f}, tau={-1 / p[2]:+.3f}'
                ax2.plot(t, e(t, p0), '-', label=f'Initial ({strc0})')
                ax2.plot(t, e(t, p), '--', label=f'fFit ({strc})')
            ax2.legend()

    return p


def fitd2(t, v, plot=False):
    """
    Fits a double-exponential ``y = a + b0 * exp(c0 * x) + b1 * exp(c1 * x)``,
    where ``b0`` and ``b1`` have the same sign, ``c0`` and ``c1`` are both
    negative, and ``c1 > c0``.

    Arguments:

    ``t``, ``v``
        The time series
    ``plot``
        Optional parameter to create a plot of the method's workings. Can be a
        boolean or an array of known (true) parameters.

    Returns an :class:`ExponentialFit`.
    """
    t, v = expfit.vet_series(t, v)

    # Estimate the dominant rate (in transformed space)
    tr = expfit.UnitSquareTransform(t, v)
    q0 = expfit.estimate_initial_single(tr.x, tr.y, vet=False)
    a0, b0, c0 = tr.detransform(q0)
    del tr, q0

    # Avoid nans etc.
    if c0 == 0:
        return expfit.ExponentialFit(t, v, (a0, b0, 0, 0, 0))

    # Catch non-decaying
    if c0 > 0:
        raise RuntimeError(
            'Initial estimate for c > 0, exponential not decaying')

    # Calculate area, to determine new b constants
    A0 = trapezoid(v - a0, t)

    # Fit double (in untransformed space)
    # Assume dominant (slowest) rate found, next will be faster
    p0 = np.array((a0, b0, c0, b0, c0), dtype=float)
    ct = expfit.DecayingConstraint()
    for i in range(1, 10):
        # Increase the difference between dominant and second exponential.
        p0[2] *= 0.707106781
        p0[4] *= 1.414213562

        # Set b constants to get same area under the curve as original estimate
        A = (p0[1] / p0[2] * (np.exp(p0[2] * t[-1]) - np.exp(p0[2] * t[0])) +
             p0[3] / p0[4] * (np.exp(p0[4] * t[-1]) - np.exp(p0[4] * t[0])))
        p0[1] = b0 * (A0 / A)
        p0[3] = b0 * (A0 / A)

        # Fit
        e = expfit.MultiExponentialError(t, v)
        with np.errstate(all='ignore'):
            r = expfit.lm(e, p0, constraint=ct)
            if plot is not False:  # pragma: no cover
                print(r)
        if r.x[4] / r.x[2] > 1.1 and r.success:
            break

    p = expfit.ExponentialFit(t, v, r.x, e, ct)

    if plot is not False:  # pragma: no cover
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(11, 7.5))
        fig.subplots_adjust(0.075, 0.06, 0.99, 0.95, wspace=0.37, hspace=0.3)
        grd = fig.add_gridspec(3, 3, width_ratios=(2, 2, 1))

        # Show data
        code = '-' if len(t) > 10 else 'x-'
        ax0 = fig.add_subplot(grd[:2, :2])
        ax0.set_xlabel('t')
        ax0.set_ylabel('v')
        ax0.plot(t, v, code, color='tab:blue', label=f'Data (n={len(t)})')

        # Show parameters
        p0 = expfit.ExponentialFit(t, v, p0)
        ax0.text(0.5, 1.015, f'Init: {p0}\n Fit: {p}',
                 transform=ax0.transAxes, ha='center', font='monospace')

        # Try showing known solution
        e = expfit.exp
        known = False
        try:
            assert len(plot) == 5
            known = True
        except (TypeError, AssertionError):
            pass
        if known:
            ax0.plot(t, e(t, (plot[0], plot[1], plot[2])), color=C1,
                     label=f'Known 1st (tau={-1 / plot[2]:.2g})',)
            ax0.plot(t, e(t, (plot[0], plot[3], plot[4])), color=C2,
                     label=f'Known 2nd (tau={-1 / plot[4]:.2g})')

        # Show fit
        label = f'rmse {np.sqrt(r.error):.4}'
        if r.success:
            label = f'Fit ({r.iterations} iter, {label})'
        else:
            label = f'Fit ({r.message}, {label})'
        ax0.plot(t, e(t, p), lw=1, color='k', label=label)

        # First exponential
        lo1, hi1 = p.ci_profile(2)
        cif1 = p.ci_fisher(2)
        tau1 = -1 / p[2]
        t1lop, t1hip = -1 / lo1[2], -1 / hi1[2]
        t1lof, t1hif = -1 / (p[2] - cif1), -1 / (p[2] + cif1)
        b = (f'Fit 1st (tau={tau1:.2g}, PL[{t1lop:.2g}, {t1hip:.2g}],'
             f' FI[{t1lof:.2g}, {t1hif:.2g}])')
        ax0.plot(t, e(t, (p[0], p[1], p[2])), lw=1, ls='--', color=D1, label=b)
        ax0.fill_between(t, e(t, (lo1[0], lo1[1], lo1[2])),
                         e(t, (hi1[0], hi1[1], hi1[2])), color=D1, alpha=0.1)
        ax0.plot(t, e(t, (lo1[0], lo1[1], lo1[2])), lw=0.4, color=D1)
        ax0.plot(t, e(t, (hi1[0], hi1[1], hi1[2])), lw=0.4, color=D1)
        ax0.plot(t, e(t, lo1), 'tab:green', ls='--', lw=0.4)
        ax0.plot(t, e(t, hi1), 'tab:green', ls='--', lw=0.4)

        # Second exponential
        lo2, hi2 = p.ci_profile(4)
        cif2 = p.ci_fisher(4)
        tau2 = -1 / p[4]
        t2lop, t2hip = -1 / lo2[4], -1 / hi2[4]
        t2lof, t2hif = -1 / (p[4] - cif2), -1 / (p[4] + cif2)
        b = (f'Fit 2nd (tau={tau2:.2g} P[{t2lop:.2g}, {t2hip:.2g}],'
             f' FI[{t2lof:.2g}, {t2hif:.2g}])')
        ax0.plot(t, e(t, (p[0], p[3], p[4])), lw=1, ls='--', color=D2, label=b)
        ax0.fill_between(t, e(t, (lo2[0], lo2[3], lo2[4])),
                         e(t, (hi2[0], hi2[3], hi2[4])), color=D2, alpha=0.1)
        ax0.plot(t, e(t, (lo2[0], lo2[3], lo2[4])), lw=0.4, color=D2)
        ax0.plot(t, e(t, (hi2[0], hi2[3], hi2[4])), lw=0.4, color=D2)
        ax0.plot(t, e(t, lo2), 'tab:green', ls='--', lw=0.4)
        ax0.plot(t, e(t, hi2), 'tab:green', ls='--', lw=0.4)

        # Finalise main panel
        ax0.legend(framealpha=1, ncol=2)

        # Show single exponential estimate
        ax1 = fig.add_subplot(grd[2, 0])
        ax1.set_xlabel('t')
        ax1.set_ylabel('v')
        ax1.plot(t, v, code, color='tab:blue', label='Data')
        ax1.plot(t, e(t, (a0, b0, c0)), 'k--', lw=1.5,
                 label=f'Init. single ($\\tau$={-1 / c0:.2g})')
        ax1.plot(t, e(t, p0), 'k:', lw=1.5,
                 label='Init. double')
        ax1.legend(frameon=False)

        # Show final fit residuals
        ax2 = fig.add_subplot(grd[2, 1])
        ax2.set_xlabel('t')
        ax2.set_ylabel('Residuals')
        ax2.plot(t, v - e(t, p))

        # Show MSE profile for tau 1
        ax4 = fig.add_subplot(grd[0, 2])
        ax4.set_xlabel('tau 1')
        ax4.set_ylabel('MSE')
        values, errors = p.profile(2, lo1[2], hi1[2])
        ax4.plot(-1 / values, errors, label='Profile')
        ax4.axvline(tau1, color='gray', zorder=1)
        ax4.axvline(t1lop, color='tab:blue', lw=1, ls='--')
        ax4.axvline(t1hip, color='tab:blue', lw=1, ls='--')

        # Show FIM approximation for tau1
        q = 0.5 / np.diag(np.linalg.inv(r.hes))
        x = np.linspace(-cif1, cif1, 100)
        ax4.plot(-1 / (p[2] + x), r.error + q[2] * x**2, label='FI')
        ax4.legend(loc=(0, 1.01), ncols=2, frameon=False, handlelength=1.5)
        ax4.axvline(t1lof, color='tab:orange', lw=1, ls='--')
        ax4.axvline(t1hif, color='tab:orange', lw=1, ls='--')

        # Show MSE profile for tau 2
        ax5 = fig.add_subplot(grd[1, 2])
        ax5.set_xlabel('tau 2')
        ax5.set_ylabel('MSE')
        values, errors = p.profile(4, lo2[4], hi2[4])
        ax5.plot(-1 / values, errors)
        ax5.axvline(-1 / p[4], color='gray', zorder=1)
        ax5.axvline(t2lop, color='tab:blue', lw=1, ls='--')
        ax5.axvline(t2hip, color='tab:blue', lw=1, ls='--')

        # Show FIM approximation for tau2
        x = np.linspace(-cif2, cif2, 100)
        ax5.plot(-1 / (p[4] + x), r.error + q[4] * x**2)
        ax5.axvline(t2lof, color='tab:orange', lw=1, ls='--')
        ax5.axvline(t2hif, color='tab:orange', lw=1, ls='--')

        # Show error comparison with known
        if known:
            ax3 = fig.add_subplot(grd[2, 2])
            plot_vs_true(ax3, p, plot)
            fig.align_ylabels((ax3, ax4, ax5))
        else:
            fig.align_ylabels((ax4, ax5))

    return p


def plot_vs_true(ax, fit, known, padding=0.25,
                 evaluations=200):  # pragma: no cover
    """
    Plots the MSE between a ``found`` and ``known`` (true) value.
    """
    found, known = np.array(fit), np.array(known)
    e = fit.error()
    s = np.linspace(-padding, 1 + padding, evaluations)
    r = known - found
    x = [found + sj * r for sj in s]
    y = [e.mse(i) for i in x]
    ax.plot(s, y, color='green')
    ax.axvline(0, color='#1f77b4', label='Found')
    ax.axvline(1, color='#7f7f7f', label='True')
    emax = fit.mse_cutoff()
    ax.axhline(emax, color='tab:red', lw=1, ls=':', label='CI cut-off')
    ax.set_ylabel('MSE')
    ax.legend()


def trapezoid(y, x):
    """ For compatibility with numpy < 2 """
    try:
        return np.trapezoid(y, x)
    except AttributeError:  # pragma: no cover
        return np.trapz(y, x)

