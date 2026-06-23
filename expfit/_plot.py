#
# Shared debugging plots.
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import numpy as np

import expfit


colors = [
    ('tab:red', '#961b1c'),
    ('tab:purple', '#683e8f'),
    ('tab:orange', '#bc5800'),
    ('tab:pink', '##c92998'),
    ('tab:brown', '##623c34'),
]


def scale_lightness(color, scale=0.7):
    """
    Takes a color in matplotlib format, scales its lightness by ``scale``, and
    returns a hex code.
    """
    import colorsys
    import matplotlib
    r, g, b = matplotlib.colors.ColorConverter.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = min(1, l * scale)
    return matplotlib.colors.to_hex(colorsys.hls_to_rgb(h, l, s))


def nth(i):
    """ Converts 0 to '1st', 1 to '2d' etc. """
    if i == 0:
        return '1st'
    return f'{1 + i}d' if i < 3 else f'{1 + i}th'


def exp_plot(t, p):
    """
    Plots an exponential, and its individual components.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 7.5))
    fig.subplots_adjust(0.075, 0.06, 0.99, 0.95)
    ax = fig.add_subplot()
    ax.plot(t, expfit.exp(t, p), 'k', label='Combined')

    # Calculate contribution to area of each
    d = (len(p) - 1) // 2
    A = np.array(
        [np.abs(expfit.area(t, p[1 + 2 * i:3 + 2 * i:])) for i in range(d)])
    Ar = 100 * A / np.sum(A)

    for i in range(d):
        ax.plot(t, expfit.exp(t, (p[0], p[1 + 2 * i], p[2 + 2 * i])),
                label=f'{nth(i)}, A={A[i]:.3} ({Ar[i]:.3}%)')
    ax.legend()


def initial_estimate_plot(x, y, estimate):
    """
    Creates a plot of the initial single estimate routine, showing the segment
    selection and refinement.

    Arguments:

    ``x``, ``y``
        The time series.
    ``estimate``
        The :class:`expfit.SingleExponentialEstimate`. Must have been obtained
        with the full extra properties.

    Returns a tuple ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11, 7.5))
    fig.subplots_adjust(0.075, 0.06, 0.99, 0.95)
    ax = fig.add_subplot()

    # Show zoomed region
    if estimate.region is not None:
        i, j = estimate.region
        ax.axvspan(x[i], x[j - 1], color='#eee')

    # Show data and estimate
    ax.plot(x, y, 's-' if len(x) < 50 else '-', label=f'Data (n={len(x)})')
    ax.plot(x, expfit.exp(x, estimate), '--',
            label=f'Initial estimate ({estimate})')

    # Show shrinking segments
    for log, color in ((estimate.log1, 'k'), (estimate.log2, 'r')):
        for ls, msg in log[:-1]:
            ax.plot(ls.x, ls.y, color=color, ls='-', label=msg)
            ax.plot(ls.mu_x, ls.mu_y, 's', color=color, fillstyle='full')
        ls, msg = log[-1]
        ax.plot(ls.x, ls.y, color=color, ls=':', label=msg)
        ax.plot(ls.mu_x, ls.mu_y, 's', color=color, fillstyle='none')

    ax.legend()
    return fig, ax


def fit1_plot(t, v, tr, r, p, q0, pt=None):
    """
    Creates a plot of a single-exponential fit, highlighting the initial
    estimate.

    Arguments:

    ``t``, ``v``
        The untransformed time series.
    ``tr``
        A :class:`UnitSquareTransform` on ``(t, v)``.
    ``r``
        An :class:`LMResult`.
    ``p``
        An :class:`ExponentialFit` result, in tau-form.
    ``q0``
        A :class:`SingleExponentialEstimate`, in unit transformed space and
        c-form.
    ``pt``
        An optional parameter vector with the known solution, in untransformed
        space and tau-form.

    Returns a tuple ``(fig, (ax0, ax1, ax2))``.
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9, 7.5))
    fig.subplots_adjust(0.11, 0.06, 0.995, 0.995, wspace=0.3, hspace=0.44)

    # Show transformed data, intial estimate, and fit
    ax0 = fig.add_subplot(2, 1, 1)
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')

    ls, color = ('-', '#92cc92') if len(tr.x) > 10 else ('x-', 'tab:green')
    ax0.plot(tr.x, tr.y, ls, color=color, label='Transformed data')

    f1 = lambda p: ', '.join(f'{i:.3}' for i in p)
    ax0.plot(tr.x, expfit.expc(tr.x, q0), '-', label=f'Initial ({f1(q0)})')
    if q0.log1 is not None and len(q0.log1) > 0:
        lsfit = q0.log1[-1][0]
        ax0.plot(lsfit.x, lsfit.y, 'k', zorder=4)
        ax0.plot(lsfit.mu_x, lsfit.mu_y, 'ks', zorder=4)
    if q0.log2 is not None and len(q0.log2) > 0:
        lsfit = q0.log2[-1][0]
        ax0.plot(lsfit.x, lsfit.y, 'r', zorder=4)
        ax0.plot(lsfit.mu_x, lsfit.mu_y, 'rs', zorder=4)

    label = f'RMSE {np.sqrt(r.error):.4}'
    label = (f'Fit ({f1(r.x)}), {r.iterations} iter, {label}' if r.success else
             f'Fit ({f1(r.x)}), {r.message}, {label}')
    ax0.plot(tr.x, expfit.expc(tr.x, r.x), '--', label=label)
    ax0.legend()

    # Show numerical results
    f2 = lambda p: ' '.join(f'{i:+.5e}' for i in p)
    p0 = tr.detransform(q0)
    p0[2] = -1 / p0[2]
    lines = [f'Transformed Init: {f2(q0)}', f'             Fit:  {f2(r.x)}',
             f'Real-world  Init: {f2(p0)}', f'             Fit:  {f2(p)}']
    ax0.text(0.75, -0.38, '\n'.join(lines), transform=ax0.transAxes,
             ha='right', font='monospace')

    # Show the residuals for initial estimate and fit
    ax1 = fig.add_subplot(2, 2, 3)
    ax1.set_xlabel('t')
    ax1.set_ylabel('Residuals')
    ax1.plot(t, v - expfit.exp(t, p0), label='Initial')
    rmse = expfit.rmse(t, v, p)
    ax1.plot(t, v - expfit.exp(t, p), label=f'Fit, RMSE {rmse:.5}')
    if pt is not None:
        rmse = expfit.rmse(t, v, pt)
        ax1.plot(t, v - expfit.exp(t, pt), ':', label=f'True, RMSE {rmse:.5}')
    ax1.legend()

    # Show detransformed initial and fit
    ax2 = fig.add_subplot(2, 2, 4)
    ax2.set_xlabel('t')
    ax2.set_ylabel('v')
    label = 'Original data'
    with np.errstate(divide='ignore'):
        if pt is not None:
            label = f'{label}, tau={pt[2]:+.3f}'
        ax2.plot(t, v, ls, color=color, label=label)
        ax2.plot(t, expfit.exp(t, p0), '-', label=f'Initial, tau={p0[2]:+.3f}')
        ax2.plot(t, expfit.exp(t, p), '--', label=f'Fit, tau={p[2]:+.3f}')
    ax2.legend()

    return fig, (ax0, ax1, ax2)


def tau_plot(t, v, r, p, p0, pe=None, pt=None):
    """
    Creates a plot of a multi-exponential (decaying) fit, highlighting the time
    constants.

    Arguments:

    ``t``, ``v``
        The time series.
    ``r``
        An :class:`LMResult`.
    ``p``
        An :class:`ExponentialFit` for the obtained result.
    ``p0``
        An :class:`ExponentialFit` for the initial guess.
    ``pe``
        An optional :class:`ExponentialFit` for the initial single exponential
        estimate.'
    ``pt``
        An optional :class:`ExponentialFit` for the true parameters.

    Returns a tuple ``(fig, (main_axes, right_axes, tau_axes))``
    """
    d = (len(p) - 1) // 2

    # Can map known to found?
    known_to_found = (pt is not None and len(pt) == len(p))

    # Create figure and grids
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11, 7.5))
    fig.subplots_adjust(0.075, 0.06, 0.99, 0.95, wspace=0.22, hspace=0.25)
    gr1 = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(3, 1))
    gr2 = gr1[0, 1].subgridspec(3 if known_to_found else 2, 1)
    gr3 = gr1[1, :].subgridspec(1, d)

    # Show data
    code = '-' if len(t) > 10 else 'x-'
    ax0 = fig.add_subplot(gr1[0, 0])
    ax0.set_xlabel('t')
    ax0.set_ylabel('v')
    ax0.plot(t, v, code, color='tab:blue', label=f'Data (n={len(t)})')

    # Try showing known solution
    e = expfit.exp
    if pt is not None:
        for i in range(d):
            pc = (pt[0], pt[1 + 2 * i], pt[2 + 2 * i])
            ax0.plot(t, e(t, pc), color=colors[i][0],
                     label=f'Known {nth(i)} ($\\tau$={pt[2 + 2 * i]:.3g})',)

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

        if pt is not None:
            ax.axvline(pt[j], color='k', ls='--', label='Known')

        ax.legend(loc=(0, 1.01), ncols=3, frameon=False, handlelength=1.5)
        tau_axes.append(ax)

    # Finalise main panel
    ax0.legend(framealpha=1, ncol=2)

    # Show initial guess
    ax1 = fig.add_subplot(gr2[0])
    ax1.set_xlabel('t')
    ax1.set_ylabel('v')
    ax1.plot(t, v, code)
    if pe is not None:
        ax1.plot(t, e(t, pe), 'k--', lw=1.5,
                 label=f'Single, $\\tau$={pe[2]:.3g}')
    ax1.plot(t, e(t, p0), '-', lw=1, label='Initial')
    ax1.legend(frameon=False)

    # Show final fit residuals
    ax2 = fig.add_subplot(gr2[1])
    ax2.set_xlabel('t')
    ax2.set_ylabel('Residuals')
    ax2.plot(t, v - e(t, p))
    info_axes = [ax1, ax2]

    # Show error comparison with known
    if known_to_found:
        ax3 = fig.add_subplot(gr2[2])
        info_axes.append(ax3)

        found, known = np.array(p), np.asarray(pt)
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


def opt_plot(log, previous=None):
    """
    Creates a plot of the :meth:`lm` method's progress, geared towards
    exponentials.

    Arguments:

    ``log``
        A list where each entry contains information about a successful
        iteration. Each entry is formed as ``p, mse, alpha`` where ``p`` is the
        parameter vector, ``mse`` its score, and ``alpha`` is an optimiser
        variable (see :meth:`lm`).
    ``previous``
        A tuple returned by a previous call to ``opt_plot``.

    Returns a tuple containing the figure, axes, and selected parts of the
    plot. This can be passed back in as ``previous`` to show the result of
    multiple optimisations in the same figure.
    """
    # Number of exponential components
    d = (len(log[0][0]) - 1) // 2

    # Create or re-use figure and axes
    if previous is None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(11, 7.5))
        fig.subplots_adjust(
            0.075, 0.06, 0.99, 0.95, wspace=0.22, hspace=0.4)
        grid = fig.add_gridspec(3, d - 1, height_ratios=(1, 3, 3))

        grd2 = grid[0, :].subgridspec(1, 3)
        axa = fig.add_subplot(grd2[0])
        axa.set_xlabel('Iterations')
        axa.set_ylabel('a')

        axm = fig.add_subplot(grd2[1])
        axm.set_xlabel('Iterations')
        axm.set_ylabel('MSE')
        axm.set_yscale('log')

        axl = fig.add_subplot(grd2[2])
        axl.set_xlabel('Iterations')
        axl.set_ylabel('Alpha')
        axl.set_yscale('log')

        axb, axc = [], []
        for i in range(d - 1):
            ax = fig.add_subplot(grid[1, i])
            axb.append(ax)
            ax.set_xlabel(f'b{1 + i}')
            ax.set_ylabel(f'b{2 + i}')

            ax = fig.add_subplot(grid[2, i])
            axc.append(ax)
            ax.set_xlabel(f'c{1 + i}')
            ax.set_ylabel(f'c{2 + i}')
    else:
        # Re-use an existing figure and axes
        fig, axa, axb, axc, axm, axl, diagonals = previous

        # Remove the diagonals, which were based on x-limits likely to change
        for line in diagonals:
            line.remove()

    # Isolate parts of log
    a = [row[0][0] for row in log]
    b = [[row[0][1 + 2 * i] for row in log] for i in range(d)]
    c = [[row[0][2 + 2 * i] for row in log] for i in range(d)]
    m = [row[1] for row in log]
    l = [row[2] for row in log]

    n = len(a)
    import matplotlib
    cmap = matplotlib.colormaps['jet']
    norm = matplotlib.colors.Normalize(0, n)
    cols = [cmap(norm(j)) for j in range(n)]

    axa.plot(a, '-')
    axm.plot(m, '-')
    axl.plot(l, '-')
    for j in range(n):
        axa.plot(j, a[j], 's', color=cols[j])
        axm.plot(j, m[j], 's', color=cols[j])
        axl.plot(j, l[j], 's', color=cols[j])

    for i in range(d - 1):
        axb[i].plot(b[i], b[i + 1], '-')
        axc[i].plot(c[i], c[i + 1], '-')
        for j in range(len(a)):
            axb[i].plot(b[i][j], b[i + 1][j], 's', color=cols[j])
            axc[i].plot(c[i][j], c[i + 1][j], 's', color=cols[j])

    diagonals = []
    for i in range(d - 1):
        x = axc[i].get_xlim()
        diagonals.append(axc[i].plot(x, x, '#ccc', ls='--', lw=1)[0])

    # Pass items back to allow repeated optimisations to be plotted in one fig
    return fig, axa, axb, axc, axm, axl, diagonals


def sigma_plot(t, v, x, y, r, sigma):
    """
    Plot of the noise level estimate

    Arguments:

    ``t``, ``v``
        The time series
    ``x``, ``y``
        A segment of the time series to which an exponential was fitted
    ``r``
        The residuals of ``y``, after subtraction an exponential
    ``sigma``
        The estimated standard deviation

    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 9))
    grid = fig.add_gridspec(3, 2)

    ax = fig.add_subplot(grid[0, :])
    ax.set_xlabel('t')
    ax.set_ylabel('v')
    ax.plot(t, v, label='Data')
    ax.plot(x, y, label='Used segment')
    ax.legend()

    ax = fig.add_subplot(grid[1:, 0])
    ax.set_xlabel('t')
    ax.plot(x, r, label=f'Residuals, sigma={sigma:.3}')
    ax.legend()

    ax = fig.add_subplot(grid[1:, 1])
    ax.set_xlabel('Residuals')
    ax.hist(r, bins='auto', density=True)
    var = sigma**2
    hx = np.linspace(np.min(r), np.max(r), 99)
    hy = 1 / np.sqrt(2 * np.pi * var) * np.exp(-hx**2 / (2 * var))
    ax.plot(hx, hy, label='Normal with same sigma')
    ax.legend()
    plt.show()
