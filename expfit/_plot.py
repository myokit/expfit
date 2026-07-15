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
    ('tab:pink', '#c92998'),
    ('tab:brown', '#623c34'),
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


def set_lightness(color, lightness):
    """
    Takes a color in matplotlib format, scales its lightness by ``scale``, and
    returns a hex code.
    """
    import colorsys
    import matplotlib
    r, g, b = matplotlib.colors.ColorConverter.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return matplotlib.colors.to_hex(colorsys.hls_to_rgb(h, lightness, s))


def nth(i):
    """ Converts 0 to '1st', 1 to '2d' etc. """
    if i == 0:
        return '1st'
    return f'{1 + i}d' if i < 3 else f'{1 + i}th'


def expd_plot(x, p):
    """
    Plots a decaying exponential, and its individual components.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 7.5))
    fig.subplots_adjust(0.075, 0.06, 0.99, 0.95)
    ax = fig.add_subplot()

    # Calculate contribution to area of each, and best offset to show
    d = (len(p) - 1) // 2
    A = np.zeros(d)
    offset = p[0]
    for i in range(d):
        b, c = p[1 + 2 * i], p[2 + 2 * i]
        A[i] = b / c * (np.exp(c * x[-1]) - np.exp(c * x[0]))
        offset += b * np.exp(c * x[0])
    Ar = 100 * A / np.sum(A)

    # Show components
    for i in range(d):
        b, c = p[1 + 2 * i], p[2 + 2 * i]
        a = offset - b * np.exp(c * x[0])
        ax.plot(x, expfit.expd(x, (a, b, c)),
                label=f'{nth(i)}, A={A[i]:.3} ({Ar[i]:.3}%)')

    # Add combination
    ax.plot(x, expfit.expd(x, p), 'k', label='Combined')
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
        with ``full=True``.

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

    # Show data
    ax.plot(x, y, 's-' if len(x) < 50 else '-', label=f'Data (n={len(x)})')

    # Show shrinking segments
    for log, color in ((estimate.log1, 'k'), (estimate.log2, 'r')):
        for ls in log:
            msg = f'Slope {ls.slope:.3}, n={ls.n}'
            ax.plot(ls.x, ls.y, color=color, ls='-', label=msg)
            ax.plot(ls.mu_x, ls.mu_y, 's', color=color, fillstyle='full')

    # Show estimate
    ax.plot(x, expfit.exp1(x, estimate), ls='--',
            label=f'Initial estimate ({estimate})')

    ax.legend()
    return fig, ax


def initial_opposing_plot(x, y, isplit, p0, p1):
    """
    Creates a plot of an initial estimate for opposing decaying exponentials.

    Arguments:

    ``x``, ``y``
        The time series.
    ``isplit``
        The index in the time series where the split is made.
    ``p0``
        The fit to the dominant (slowest) exponential.
    ``p1``
        The fit to the faster (initial) exponential.

    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 9))

    fig.subplots_adjust(0.07, 0.07, 0.99, 0.99)

    # Data and split
    ax = fig.add_subplot()
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.plot(x, y, 's-' if len(x) < 50 else '-', label='Data')
    ax.axvspan(x[0], x[isplit], color='#ccc')
    ax.axvline(x[isplit], color='#888', lw=1, label='Split')
    ax.set_xlim(x[0], x[-1])

    # Dominant
    ax.plot(x, expfit.exp1(x, p0), '--', label='Dominant')

    # Secondary
    a0, b0, c0 = p0
    a1, b1, c1 = p1
    ax.plot(x[:isplit], y[:isplit] - expfit.exp1(x[:isplit], (0, b0, c0)),
            color='navy',
            label='Data with dominant subtracted (offset adjusted)')
    ax.plot(x, expfit.exp1(x, (a0, b1, c1)), '--',
            label='Second (offset adjusted)')

    # Final result
    ax.plot(x, expfit.expd(x, (a0, b0, -1 / c0, b1, -1 / c1)),
            label='Combined')
    ax.legend()
    return fig, ax


def fit1_plot(xy, q0, r, xy_org, p, pt=None):
    """
    Creates a plot of a single-exponential fit, highlighting the initial
    estimate.

    Arguments:

    ``xy``
        A, possibly transformed, time series.
    ``q0``
        A :class:`SingleExponentialEstimate`, obtained on ``xy``.
    ``r``
        An :class:`LMResult`, obtained on ``xy``.
    ``xy_org``
        An untransformed time series.
    ``p``
        An :class:`ExponentialFit` result, in untransformed (``xy_org``) space.
    ``pt``
        An optional parameter vector with the known solution, in untransformed
        (``xy_org``) space.

    Returns a tuple ``(fig, (ax0, ax1, ax2))``.
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9, 7.5))
    fig.subplots_adjust(0.11, 0.06, 0.995, 0.995, wspace=0.3, hspace=0.44)

    # Show transformed data, intial estimate, and fit
    ax0 = fig.add_subplot(2, 1, 1)
    ax0.set_xlabel('x (transformed)')
    ax0.set_ylabel('y (transformed)')

    # Transformed data
    x, y = xy
    ls, color = ('-', '#92cc92') if len(x) > 10 else ('x-', 'tab:green')
    ax0.plot(x, y, ls, color=color, label='Transformed data')

    # Initial estimate and selected segments
    f1 = lambda p: ', '.join(f'{i:.3}' for i in p)
    rmse_q0 = expfit.rmse1(x, y, q0)
    ax0.plot(x, expfit.exp1(x, q0), '-',
             label=f'Initial ({f1(q0)}), RMSE {rmse_q0:.4}')
    if q0.log1 is not None and len(q0.log1) > 0:
        lsfit = q0.log1[-1]
        ax0.plot(lsfit.x, lsfit.y, 'k')
        ax0.plot(lsfit.mu_x, lsfit.mu_y, 'ks')
    if q0.log2 is not None and len(q0.log2) > 0:
        lsfit = q0.log2[-1]
        ax0.plot(lsfit.x, lsfit.y, 'r')
        ax0.plot(lsfit.mu_x, lsfit.mu_y, 'rs')

    # Fit
    label = f'RMSE {np.sqrt(r.error):.4}'
    label = (f'Fit ({f1(r.x)}), {r.iterations} iter, {label}' if r.success else
             f'Fit ({f1(r.x)}), {r.message}, {label}')
    ax0.plot(x, expfit.exp1(x, r.x), '--', label=label)
    ax0.legend()

    # Show numerical results
    f2 = lambda p: ' '.join(f'{i:+.5e}' for i in p)
    try:
        p0 = xy.detransform(q0)
    except AttributeError:
        p0 = q0
    lines = [f'Transformed Init: {f2(q0)}', f'             Fit:  {f2(r.x)}',
             f'Real-world  Init: {f2(p0)}', f'             Fit:  {f2(p)}']
    ax0.text(0.75, -0.38, '\n'.join(lines), transform=ax0.transAxes,
             ha='right', font='monospace')

    # Show detransformed residuals for initial estimate, fit, and true
    x, y = xy_org
    ax1 = fig.add_subplot(2, 2, 3)
    ax1.set_xlabel('x')
    ax1.set_ylabel('Residuals')
    rmse_p0 = expfit.rmse1(x, y, p0)
    ax1.plot(x, y - expfit.exp1(x, p0), label=f'Initial, RMSE {rmse_p0:.5}')
    rmse_p = expfit.rmse1(x, y, p)
    ax1.plot(x, y - expfit.exp1(x, p), label=f'Fit, RMSE {rmse_p:.5}')
    if pt is not None:
        rmse_pt = expfit.rmse1(x, y, pt)
        ax1.plot(
            x, y - expfit.exp1(x, pt), ':', label=f'True, RMSE {rmse_pt:.5}')
    ax1.legend()

    # Show detransformed initial, fit, and true
    ax2 = fig.add_subplot(2, 2, 4)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    label = 'Original data'
    if pt is not None:
        label = f'{label}, c={pt[2]:.3f}'
    ax2.plot(x, y, ls, color=color, label=label)
    ax2.plot(x, expfit.exp1(x, p0), '-', label=f'Initial, c={p0[2]:.3f}')
    ax2.plot(x, expfit.exp1(x, p), '--', label=f'Fit, c={p[2]:.3f}')
    ax2.legend()

    return fig, (ax0, ax1, ax2)


def tau_plot(xy, p0, r, p, pt=None):
    """
    Creates a plot of a multi-exponential (decaying) fit, highlighting the time
    constants.

    Arguments:

    ``xy``
        A :class:`expfit.TimeSeries`.
    ``p0``
        An array containing the initial guess.
    ``r``
        An :class:`LMResult`. This may be on a transformed parameter space:
        only the optimiser details are used.
    ``p``
        An :class:`ExponentialFit` for the obtained result.
    ``pt``
        An optional array with the true parameters.

    Returns a tuple ``(fig, (main_axes, right_axes, tau_axes))``
    """
    x, y = xy
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
    code = '-' if len(x) > 10 else 'x-'
    ax0 = fig.add_subplot(gr1[0, 0])
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    ax0.plot(x, y, code, color='tab:blue', label=f'Data (n={len(x)})')

    # Try showing known solution
    e = expfit.expd
    if pt is not None:
        for i in range(d):
            pc = (pt[0], pt[1 + 2 * i], pt[2 + 2 * i])
            ax0.plot(x, e(x, pc), color=colors[i][0],
                     label=f'Known {nth(i)} ($\\tau$={pt[2 + 2 * i]:.3g})',)

    # Show fit
    if r is None:
        label = 'Fit'
    elif r.success:
        label = f'Fit ({r.iterations} iter, rmse {np.sqrt(r.error):.4})'
    else:
        label = f'Fit ({r.message}, rmse {np.sqrt(r.error):.4})'
    ax0.plot(x, e(x, p), lw=1, color='k', label=label)

    # Show parameters
    p0 = expfit.ExponentialFit(p0)   # TODO
    ax0.text(0.5, 1.015, f'Init: {p0}\n Fit: {p}',
             transform=ax0.transAxes, ha='center', font='monospace')

    # Store y limits, in case PL messed them up
    ax0_ylims_a = ax0.get_ylim()

    # Components
    tau_axes = []
    for i in range(d):
        j = 2 + 2 * i
        flo, fhi = p.ci_fisher(j)
        plo, phi = p.ci_profile(j)
        c = colors[i][1]

        # Show component on main axes
        b = f'Fit {nth(i)} ($\\tau$={p[j]:.2g}, FI[{flo:.3g}, {fhi:.3g}]'
        b = f'{b}, PL[failed' if plo is None else f'{b}, PL[{plo[j]:.3g}'
        b = f'{b}, failed])' if phi is None else f'{b}, {phi[j]:.3g}])'
        pc = (p[0], p[1 + 2 * i], p[2 + 2 * i])
        ax0.plot(x, e(x, pc), lw=1, ls='--', color=c, label=b)

        # Show profile on dedicated axes
        ax = fig.add_subplot(gr3[0, i])
        ax.set_xlabel(f'Tau {1 + i}')
        ax.set_ylabel('MSE')

        # Profile log-likelihood (MSE)
        lo = p[j] + 0.5 * (flo - p[j]) if plo is None else plo[j]
        hi = p[j] + 0.5 * (fhi - p[j]) if phi is None else phi[j]
        values, errors, solutions = p.profile(j, lo, hi, solutions=True)
        ax.plot(values, errors, label='Profile')
        ax.axvline(p[j], color='gray')
        if plo is not None:
            ax.axvline(plo[j], color='tab:blue', lw=1, ls='--')
        else:
            ax.plot(values[0], errors[0], 'x', color='tab:blue')
        if phi is not None:
            ax.axvline(phi[j], color='tab:blue', lw=1, ls='--')
        else:
            ax.plot(values[-1], errors[-1], 'x', color='tab:blue')

        # Show CI on main axes
        cl = set_lightness(c, 0.9)
        for pc in solutions:
            pc = (pc[0], pc[1 + 2 * i], pc[2 + 2 * i])
            ax0.plot(x, e(x, pc), color=cl, zorder=0.5)
        # Show solutions at edges
        if plo is not None:
            pclo = (plo[0], plo[1 + 2 * i], plo[2 + 2 * i])
            ax0.plot(x, e(x, pclo), lw=0.4, color=c)
        if phi is not None:
            pchi = (phi[0], phi[1 + 2 * i], phi[2 + 2 * i])
            ax0.plot(x, e(x, pchi), lw=0.4, color=c)

        # FIM approximation
        fx = np.linspace(flo, fhi, 100)
        q = 0.5 / np.diag(np.linalg.inv(p.hes()))
        ax.plot(fx, p.mse() + q[j] * (fx - p[j])**2, 'tab:orange', label='FI')
        ax.axvline(flo, color='tab:orange', lw=1, ls='--')
        ax.axvline(fhi, color='tab:orange', lw=1, ls='--')

        # CI cut-off
        lo, hi = p.mse(), p.mse_cutoff()
        ax.axhline(hi, color='k', lw=1, ls=':')
        pad = 0.05 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)

        # True value, if known
        if pt is not None:
            ax.axvline(pt[j], color='k', ls='--', label='Known')

        ax.legend(loc=(0, 1.01), ncols=3, frameon=False, handlelength=1.5)
        tau_axes.append(ax)

    # Restore y limits, if required
    ax0_ylims_b = ax0.get_ylim()
    ra, rb = ax0_ylims_a[1] - ax0_ylims_a[0], ax0_ylims_b[1] - ax0_ylims_b[0]
    if abs(rb / ra) > 10:
        ax0.set_ylim(ax0_ylims_a)

    # Finalise main panel
    ax0.legend(framealpha=1, ncol=2)

    # Show initial guess
    ax1 = fig.add_subplot(gr2[0])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.plot(x, y, code)
    ax1.plot(x, e(x, p0), '-', lw=1, label='Initial')
    ax1.legend(frameon=False)

    # Show final fit residuals
    ax2 = fig.add_subplot(gr2[1])
    ax2.set_xlabel('x')
    ax2.set_ylabel('Residuals')
    ax2.plot(x, y - e(x, p))
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
        ex = [found + sj * r for sj in s]
        ey = [e.mse(i) for i in ex]
        ax3.plot(s, ey, color='green')
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
        if d == 1:
            grid = fig.add_gridspec(3, 1, height_ratios=(1, 3, 3))
        else:
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
        if d == 1:
            ax = fig.add_subplot(grid[1:, 0])
            axb.append(ax)
            ax.set_xlabel('b1')
            ax.set_ylabel('c1')
        else:
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
        if d > 1:
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

    diagonals = []
    if d == 1:
        axb[0].plot(b[0], c[0], '-')
        for j in range(len(a)):
            axb[0].plot(b[0][j], c[0][j], 's', color=cols[j])
    else:
        for i in range(d - 1):
            axb[i].plot(b[i], b[i + 1], '-')
            axc[i].plot(c[i], c[i + 1], '-')
            for j in range(len(a)):
                axb[i].plot(b[i][j], b[i + 1][j], 's', color=cols[j])
                axc[i].plot(c[i][j], c[i + 1][j], 's', color=cols[j])

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
