#
# Single and multi-expontial fits
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import numpy as np

import expfit


colors = [
    ('tab:red', '#961b1c'),
    ('tab:purple', '#5b3383'),
]
# '#1f701f'


def fit1(t, v, plot=False):
    """
    Fits an exponential ``a + b * exp(-t / tau)`` to the time series
    ``(t, v)``, returning ``(a, b, tau)``

    Example::

        t = np.linspace(0, 1, 100)
        v = 3 + 2 * np.exp(-t / 0.2) + np.random.normal(0, 1, size=len(t))
        a, b, tau = expfit.fit_single(t, v)
        print(a, b, tau)

    Arguments:

    ``t``, ``v``
        The time series
    ``plot``
        Optional parameter to create a plot of the method's workings.

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
    q0 = expfit.estimate_initial_single(tr.x, tr.y, axes=ax0, vet=False)

    # Stop if the signal is not exponential
    if q0[1] == 0 or q0[2] == 0:
        raise expfit.NotExponentialError()

    # Fit (in transformed space)
    e = expfit.SingleExponentialError(tr.x, tr.y)
    with np.errstate(all='ignore'):
        r = expfit.lm(e, q0)
        if plot:  # pragma: no cover
            print(r)
    p = tr.detransform(r.x)
    p[2] = -1 / p[2]

    # Detransform obtained parameters, switch to tau form, create result object
    e = expfit.TauFormError(t, v)
    p = expfit.ExponentialFit(t, v, p, e)

    if plot:  # pragma: no cover
        # Detransform and rewrite in tau form
        p0 = tr.detransform(q0)
        p0[2] = -1 / p0[2]
        p0 = expfit.ExponentialFit(t, v, p0)
        q0 = list(q0)
        q0[2] = -1 / q0[2]
        q0 = expfit.ExponentialFit(tr.x, tr.y, q0)
        q = r.x
        q[2] = -1 / q[2]
        q = expfit.ExponentialFit(tr.x, tr.y, r.x)

        # Create strings for plot labels
        strest = ', '.join(f'{i:.3}' for i in q0)
        strq = ', '.join(f'{i:.3}' for i in q)
        stre = f'rmse {np.sqrt(r.error):.4}'
        if r.success:
            strfit = f'{r.iterations} iter, {stre}'
        else:
            strfit = f'{r.message}, {stre}'

        # Plot initial estimate and fit
        e = expfit.exp
        ax0.plot(tr.x, e(tr.x, q0), '-', label=f'Initial ({strest})')
        ax0.plot(tr.x, e(tr.x, q), '--', label=f'Fit ({strq}), {strfit}')
        ax0.legend()

        # Plot numerical results
        lines = [f'Transformed Init: {q0}', f'             Fit:  {q}',
                 f'Real-world  Init: {p0}', f'             Fit:  {p}']
        ax0.text(0.75, -0.38, '\n'.join(lines), transform=ax0.transAxes,
                 ha='right', font='monospace')

        # Show the residuals for initial estimate and fit
        ax1 = fig.add_subplot(2, 2, 3)
        ax1.set_xlabel('x')
        ax1.set_ylabel('Residuals (transformed)')
        ax1.plot(tr.x, tr.y - e(tr.x, q0), label='Initial')
        ax1.plot(tr.x, tr.y - e(tr.x, q), label='Fit')
        ax1.legend()

        # Show untransformed, including the original data
        ax2 = fig.add_subplot(2, 2, 4)
        ax2.set_xlabel('t')
        ax2.set_ylabel('v')
        label = 'Original data'
        with np.errstate(divide='ignore'):
            if known:
                label = f'{label} (tau={known[2]:+.3f})'
            ax2.plot(t, v, ls, color=color, label=label)
            ax2.plot(t, e(t, p0), '-', label=f'Initial (tau={p0[2]:+.3f})')
            ax2.plot(t, e(t, p), '--', label=f'fFit (tau={p[2]:+.3f})')
        ax2.legend()

    return p


def fitd2(t, v, plot=False, opt_plot=False):
    """
    Fits a decaying double-exponential to a time series.

    Returns parameters for::

        v = a + b0 * exp(-t / tau1) + b1 * exp(-t / tau2)

    where ``tau0 > tau1``.

    Arguments:

    ``t``, ``v``
        The time series
    ``plot``
        Optional parameter to create a plot showing the final results,
        including confidence intervals on the time constants.
    ``opt_plot``
        Optional parameter to create a plot of the optimisation routine.

    Returns an :class:`ExponentialFit`.
    """
    t, v = expfit.vet_series(t, v)

    # Estimate the dominant rate
    tr = expfit.UnitSquareTransform(t, v)
    q0 = expfit.estimate_initial_single(tr.x, tr.y, vet=False)
    a0, b0, c0 = tr.detransform(q0)
    del tr, q0

    # Stop if the signal is not exponential
    if b0 == 0 or c0 == 0:
        raise expfit.NotExponentialError()

    # Catch non-decaying
    if c0 > 0:
        raise expfit.NotDecayingError()

    # Fit double (in untransformed space)

    # Calculate area, to determine new b constants
    A0 = (b0 / c0) * (np.exp(c0 * t[-1]) - np.exp(c0 * t[0]))

    # Assume dominant (slowest) rate found, next will be faster
    p0 = np.array((a0, b0, c0, b0, c0), dtype=float)
    p0[1] *= 0.7    # The second exponential will contribute
    p0[2] *= 0.5    # The first c will be overestimated

    # Set up error
    npos, nneg = (2, 0) if b0 > 0 else (0, 2)
    e = expfit.MultiExponentialError(t, v, npos, nneg)

    max_iter = 10
    opt_fig = opt_plot
    for i in range(max_iter):
        # Speed up the second exponential
        p0[4] *= 1.4

        # Set b constants to get same area under the curve as original estimate
        A1 = p0[1] / p0[2] * (np.exp(p0[2] * t[-1]) - np.exp(p0[2] * t[0]))
        A2 = p0[3] / p0[4] * (np.exp(p0[4] * t[-1]) - np.exp(p0[4] * t[0]))
        p0[1] = p0[3] = b0 * (A0 / (A1 + A2))

        # Fit with transformed parameters
        q0 = e.transform(p0)
        with np.errstate(all='ignore'):
            r = expfit.lm(e, q0, plot=opt_fig)
            if plot is not False:  # pragma: no cover
                print(r)
            opt_fig = r.plot

        if np.exp(r.x[4] - r.x[2]) > 1.1 and r.success:
            break
        elif i + 1 == max_iter:  # pragma: no cover
            raise RuntimeError(
                f'Unable to find good fit after {max_iter} attempts.')

    # Detransform parameters
    et = expfit.TauFormError(t, v)
    p = expfit.ExponentialFit(t, v, e.detransform(r.x, True), et)

    if plot is not False:  # pragma: no cover
        pk = None
        try:
            assert len(plot) == 5
            pk = plot
        except (TypeError, AssertionError):
            pass
        p0 = expfit.ExponentialFit(t, v, e.detransform(q0, True), et)
        fig, (ax, iax, tax) = tau_plot(t, v, r, p, p0, pk)
        iax[0].plot(t, expfit.expc(t, (a0, b0, c0)), 'k--', lw=1.5,
                    label=f'Single ($\\tau$={-1 / c0:.3g})')
        iax[0].legend(frameon=False)

    return p


def fitd11(t, v, plot=False):
    """
    Fits a double-exponential ``y = a + b0 * exp(c0 * x) + b1 * exp(c1 * x)``,
    where ``b0`` and ``b1`` have different signs, ``c0`` and ``c1`` are both
    negative, and ``c1 > c0``.

    TODO
    TODO
    TODO
    TODO
    TODO
    TODO
    TODO

    Arguments:

    ``t``, ``v``
        The time series
    ``plot``
        Optional parameter to create a plot of the method's workings. Can be a
        boolean or an array of known (true) parameters.

    Returns an :class:`ExponentialFit`.
    """
    t, v = expfit.vet_series(t, v)

    # Perform initial estimates in transformed space
    tr = expfit.UnitSquareTransform(t, v)
    q0 = expfit.estimate_initial_opposing(tr.x, tr.y, vet=False, plot=False)
    p0 = tr.detransform(q0)
    del tr, q0

    # Catch edge cases
    if p0[1] * p0[3] >= 0:
        raise expfit.NotOpposingError()
    if p0[2] > 0 or p0[4] > 0:
        raise expfit.NotDecayingError()

    # Fit double
    npos, nneg, pos_first = 1, 1, p0[1] > 0
    e = expfit.MultiExponentialError(t, v, npos, nneg, pos_first)
    q0 = e.transform(p0)
    with np.errstate(all='ignore'):
        r = expfit.lm(e, q0)
        if plot is not False:  # pragma: no cover
            print(r)

    # Detransform parameters
    et = expfit.TauFormError(t, v)
    p = expfit.ExponentialFit(t, v, e.detransform(r.x, True), et)

    if plot is not False:  # pragma: no cover
        pt = None
        p0 = expfit.ExponentialFit(t, v, e.detransform(q0, True), et)
        try:
            assert len(plot) == 5
            pt = plot
        except (TypeError, AssertionError):
            pass
        tau_plot(t, v, r, p, p0, pt)

    return p


def tau_plot(t, v, r, p, p0, ptrue=None):  # pragma: no cover
    """
    Creates a debug plot for a bi-exponential (decaying, with equal or opposing
    signs).

    Arguments:

    ``t``, ``v``
        The time series.
    ``r``
        An :class:`LMResult`.
    ``p``
        An :class:`ExponentialFit` for the obtained result.
    ``p0``
        An :class:`ExponentialFit` for the initial guess.
    ``p0``
        An optional :class:`ExponentialFit` for the true parameters.

    Returns a tuple ``(fig, (main_axes, right_axes, tau_axes))``
    """
    d = (len(p) - 1) // 2

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11, 7.5))
    fig.subplots_adjust(0.075, 0.06, 0.99, 0.95, wspace=0.22, hspace=0.25)
    gr1 = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(3, 1))
    gr2 = gr1[0, 1].subgridspec(2 if ptrue is None else 3, 1)
    gr3 = gr1[1, :].subgridspec(1, d)

    # Show data
    code = '-' if len(t) > 10 else 'x-'
    ax0 = fig.add_subplot(gr1[0, 0])
    ax0.set_xlabel('t')
    ax0.set_ylabel('v')
    ax0.plot(t, v, code, color='tab:blue', label=f'Data (n={len(t)})')

    # Try showing known solution
    e = expfit.exp
    if ptrue is not None:
        for i in range(d):
            pc = (ptrue[0], ptrue[1 + 2 * i], ptrue[2 + 2 * i])
            ax0.plot(t, e(t, pc), color=colors[i][0],
                     label=f'Known {nth(i)} ($\\tau$={ptrue[2 + 2 * i]:.3g})',)

    # Show fit
    if r.success:
        label = f'Fit ({r.iterations} iter, rmse {np.sqrt(r.error):.4})'
    else:
        label = f'Fit ({r.message}, rmse {np.sqrt(r.error):.4})'
    ax0.plot(t, e(t, p), lw=1, color='k', label=label)

    # Show parameters
    p0 = expfit.ExponentialFit(t, v, p0)
    ax0.text(0.5, 1.015, f'Init: {p0}\n Fit: {p}',
             transform=ax0.transAxes, ha='center', font='monospace')

    # Components
    tau_axes = []
    for i in range(d):
        j = 2 + 2 * i
        flo, fhi = p.ci_fisher(j)
        try:
            profile = True
            plo, phi = p.ci_profile(j)
        except expfit.CILimitNotFound:
            profile = False

        c = colors[i][1]

        # Show component and PL CI on main axes
        b = f'Fit {nth(i)} ($\\tau$={p[j]:.2g}, FI[{flo:.3g}, {fhi:.3g}]'
        if profile:
            b = f'{b}, PL[{plo[j]:.3g}, {phi[j]:.3g}])'
        else:
            b = f'{b}, PL Failed)'
        pc = (p[0], p[1 + 2 * i], p[2 + 2 * i])
        ax0.plot(t, e(t, pc), lw=1, ls='--', color=c, label=b)
        if profile:
            pclo = (plo[0], plo[1 + 2 * i], plo[2 + 2 * i])
            pchi = (plo[0], phi[1 + 2 * i], phi[2 + 2 * i])
            ax0.fill_between(t, e(t, pclo), e(t, pchi), color=c, alpha=0.1)
            ax0.plot(t, e(t, pclo), lw=0.4, color=c)
            ax0.plot(t, e(t, pchi), lw=0.4, color=c)
        #ax0.plot(t, e(t, plo), 'tab:green', ls='--', lw=0.4)
        #ax0.plot(t, e(t, phi), 'tab:green', ls='--', lw=0.4)

        # Show profile on dedicated axes
        ax = fig.add_subplot(gr3[0, i])
        ax.set_xlabel(f'Tau {1 + i}')
        ax.set_ylabel('MSE')

        # Profile log-likelihood (MSE)
        if profile:
            values, errors = p.profile(j, plo[j], phi[j])
            ax.plot(values, errors, label='Profile')
            ax.axvline(p[j], color='gray')
            ax.axvline(plo[j], color='tab:blue', lw=1, ls='--')
            ax.axvline(phi[j], color='tab:blue', lw=1, ls='--')

        # FIM approximation
        x = np.linspace(flo, fhi, 100)
        q = 0.5 / np.diag(np.linalg.inv(p.hes()))
        ax.plot(x, p.mse() + q[j] * (x - p[j])**2, 'tab:orange', label='FI')
        ax.axvline(flo, color='tab:orange', lw=1, ls='--')
        ax.axvline(fhi, color='tab:orange', lw=1, ls='--')

        if ptrue is not None:
            ax.axvline(ptrue[j], color='k', ls='--', label='Known')

        ax.legend(loc=(0, 1.01), ncols=3, frameon=False, handlelength=1.5)
        tau_axes.append(ax)

    # Finalise main panel
    ax0.legend(framealpha=1, ncol=2)

    # Show initial guess
    ax1 = fig.add_subplot(gr2[0])
    ax1.set_xlabel('t')
    ax1.set_ylabel('v')
    ax1.plot(t, v, code, color='tab:blue')
    ax1.plot(t, e(t, p0), 'k:', lw=1.5, label='Initial')
    ax1.legend(frameon=False)

    # Show final fit residuals
    ax2 = fig.add_subplot(gr2[1])
    ax2.set_xlabel('t')
    ax2.set_ylabel('Residuals')
    ax2.plot(t, v - e(t, p))
    info_axes = [ax1, ax2]

    # Show error comparison with known
    if ptrue is not None:
        ax3 = fig.add_subplot(gr2[2])
        info_axes.append(ax3)

        found, known = np.array(p), np.asarray(ptrue)
        e = p.error()
        padding = 0.25
        s = np.linspace(-padding, 1 + padding, 100)
        r = known - found
        x = [found + sj * r for sj in s]
        y = [e.mse(i) for i in x]
        ax3.plot(s, y, color='green')
        ax3.axvline(0, color='#1f77b4')
        ax3.axvline(1, color='#7f7f7f')
        emax = p.mse_cutoff()
        ax3.axhline(emax, color='tab:red', lw=1, ls=':', label='CI cut-off')
        ax3.set_ylabel('MSE')
        ax3.set_xticks([0, 1])
        ax3.set_xticklabels(['Found', 'True'])
        ax3.legend()

    fig.align_ylabels(info_axes)
    return fig, (ax0, info_axes, tau_axes)


def nth(i):  # pragma: no cover
    """ Converts 0 to '1st', 1 to '2d' etc. """
    if i == 0:
        return '1st'
    return f'{1 + i}d' if i < 3 else f'{1 + i}th'

