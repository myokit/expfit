#
# Single expontial fits, form the basis of multi-exponential fits
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import numpy as np

import expfit


def estimate_initial_single(x, y, plot=False, axes=None, vet=True):
    """
    Estimate ``a, b, c`` in ``y = a + b * exp(c * x)`` using derivatives
    estimated from mean averages at the sides.

    The method assumes ``x`` and ``y`` are already transformed to the unit
    square.

    The method first selects two segments, one at the start of the signal and
    one near the end, and approximates them with a straight line to derive
    ``(x1, y1, dydx1)`` and ``(x2, y2, dydx2)``. It then estimates c from

        y    = a + b * exp(c * x)
        dydx = c * b * exp(c * x)

        y_1    - y_2    =     b * (exp(c * y_1) - exp(c * y_2))
        dydx_1 - dydx_2 = c * b * (exp(c * y_1) - exp(c * y_2))
        c = (dydx_1 - dydx_2) / (y_1 - y_2)

    Either segment can then be used to derive ``a`` and ``b``, from

        a = y_i - dydx_i / c
        b = (y_i - a) / np.exp(c * x_i)

    To pick a segment...
    ...
    ...
    ...
    ...

    Arguments:

    ``x``, ``y``
        A time vector and the correspond values. Assumed to be transformed onto
        the unit square.
    ``plot=False``
        Set to ``True`` to create a full debugging plot.
    ``axes=None``
        If ``matplotlib.Axes`` are passed in, the selected segments and
        estimated slopes will be drawn in.
    ``vet=True``
        Set to ``False`` to disable checks on the dimensions of ``t`` and
        ``v``. This should only be done if the input data is already vetted.

    Returns a tuple ``(a, b, c)``.
    """
    if vet:
        x_org, y_org = expfit.vet_series(x, y)
    else:
        x_org, y_org = x, y
    if len(x_org) < 3:
        raise ValueError('At least 3 points are required')

    # Obtain a result to pass back in case of failure
    abc_fail = np.mean(y_org), 0.0, 0.0

    # Transform to zoom in on the action
    tr = expfit.ZoomTransform(x_org, y_org)
    x, y = tr.x, tr.y

    # Create plot
    if plot:  # pragma: no cover
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot()
        if tr.ibounds is not None:
            i, j = tr.ibounds
            ax.axvspan(x_org[i], x_org[j - 1], color='#eee')
        ax.plot(x_org, y_org, 's-' if len(x_org) < 50 else '-')

    if plot or axes is not None:  # pragma: no cover
        def plot_line(ax, x, ls, start=True, msg=None):
            n = len(x)
            x, y = tr.detransform_series(
                np.array((x[0], x[-1])),
                np.array((ls.mu_y + ls.slope * (x[0] - ls.mu_x),
                          ls.mu_y + ls.slope * (x[-1] - ls.mu_x)))
            )
            mu_x, mu_y = tr.detransform_series(
                np.array((ls.mu_x)), np.array((ls.mu_y))
            )

            color = 'k' if start else 'r'
            label = f'Slope {ls.slope:.3} (n={n})'
            if msg is not None:
                label = f'{label}: {msg}'

            ax.plot(x, y, color=color, zorder=10, label=label)
            ax.plot(mu_x, mu_y, 's', color=color, zorder=10)

    # Return a consistent result when things go wrong
    def fail(seg1, seg2, l1, l2, msg):
        if plot:  # pragma: no cover
            plot_line(ax, seg1[0], l1, True, msg)
            plot_line(ax, seg2[0], l2, False, msg)
            ax.legend()
        return abc_fail

    # Straight line detection on t,v???
    # Maybe.

    #
    # To approximate the two derivatives, the start and end of the signal are
    # approximated linearly.
    # To find the approximations, we start with two lines covering half the
    # data each (with one point overlap in case of an odd number of points).
    # Three slopes are determined using linear least squares: s0 from the full
    # data, s1 from the signal start, s2 from the signal end.
    # - If the sign of s1 or s2 differs from s0, it is set to 0
    # - If the sign matches, an iterative shrinking procedure is started.
    #   Each iteration halves the segment, as long as
    #   - the new slope has the same sign
    #   - the new slope is increased, on the fast side, or decreased, on the
    #     slow side
    #   - The ratio of increase/decrease is between 0.8 and 1.25
    #
    def shrink(seg, ls, n_min, start=True, increasing=True):
        increasing = bool(increasing)
        x, y = seg
        n = len(x)

        r = None
        msg = None
        while n > n_min:
            # Show previous segment
            if plot:  # pragma: no cover
                plot_line(ax, x, ls, start)

            # Propose new segment
            n = max((1 + n) // 2, n_min)
            x_new, y_new = (x[:n], y[:n]) if start else (x[-n:], y[-n:])

            # Test slope sign and magnitude
            l_new = expfit.LeastSquaresFit(x_new, y_new, vet=False)
            if l_new.slope * ls.slope < 0:
                msg = 'Sign change'
                break
            if (abs(l_new.slope) >= abs(ls.slope)) != increasing:
                msg = 'Not increasing' if increasing else 'Not decreasing'
                break

            # Test near-constant rate of change
            r_new = l_new.slope / ls.slope
            if r is not None:
                rr = r / r_new
                if rr < 0.8 or rr > 1.25:
                    msg = f'Unexpected change ratio {rr:.3}'
                    break

            # Accept
            r = r_new
            x, y, ls = x_new, y_new, l_new
            if n == n_min:
                msg = 'Minimum size reached'

        # Show last segment
        if plot:  # pragma: no cover
            plot_line(ax, x, ls, start, msg=msg)

        return (x, y), ls

    # Get starting segments, and least squares fits
    m = (1 + len(x)) // 2
    seg1 = x[:m], y[:m]
    seg2 = x[-m:], y[-m:]
    l0 = expfit.LeastSquaresFit(x, y, vet=False)
    l1 = expfit.LeastSquaresFit(*seg1, vet=False)
    l2 = expfit.LeastSquaresFit(*seg2, vet=False)

    # Slopes must match full signal slope (otherwise this is either slow drift
    # or correlated noise at the flat end of the exponential, or the signal is
    # not an exponential).
    # Slopes ok? Then start shrinking
    n_min = 5
    if l0.slope * l1.slope < 0:
        l1.slope, l1.offset = 0.0, l1.mu_y
    else:
        seg1, l1 = shrink(
            seg1, l1, n_min, True, abs(l1.slope) > abs(l2.slope))

    if l0.slope * l2.slope < 0:
        l2.slope, l2.offset = 0.0, l2.mu_y
    else:
        seg2, l2 = shrink(
            seg2, l2, n_min, False, abs(l2.slope) > abs(l1.slope))

    # Show final segments on user-provided axes
    if axes is not None:  # pragma: no cover
        plot_line(axes, seg1[0], l1, True)
        plot_line(axes, seg2[0], l2, False)

    #
    # H1: Use the derived segments to estimate the parameters
    #

    # Unpack
    x1, y1, s1 = l1.mu_x, l1.mu_y, l1.slope
    x2, y2, s2 = l2.mu_x, l2.mu_y, l2.slope

    # Edge cases for c estimate
    if s1 == s2:
        return fail(seg1, seg2, l1, l2, 'Equal slopes (c=0)')
    if y1 == y2:
        return fail(seg1, seg2, l1, l2, 'Equal means (c=inf)')

    # Estimate c, as, and bs
    c = (s1 - s2) / (y1 - y2)
    a1 = y1 - (s1 / c if s1 != 0 else 0)
    a2 = y2 - (s2 / c if s2 != 0 else 0)
    b1 = (y1 - a1) * np.exp(-c * x1)
    b2 = (y2 - a2) * np.exp(-c * x2)
    am, bm = np.mean((a1, a2)), np.mean((b1, b2))

    # Use start, end, or averaged parameters, depending on RMSE
    with np.errstate(over='ignore', divide='ignore'):
        r1 = expfit.rmse_single(x, y, a1, b1, c)
        r2 = expfit.rmse_single(x, y, a2, b2, c)
        rm = expfit.rmse_single(x, y, am, bm, c)
        if rm < r1 and rm < r2:
            a, b = am, bm
            r = rm
        elif r1 < r2:
            a, b = a1, b1
            r = r1
        else:
            a, b = a2, b2
            r = r2

        # Compare with flat line
        rr = r / expfit.rmse_single(x, y, l0.mu_y, 0, 0)
    if rr > 2:
        return fail(seg1, seg2, l1, l2, 'Flat line is better fit')

    # Show initial estimate
    if plot:  # pragma: no cover
        p, q, r = tr.detransform(a, b, c)
        ax.plot(x_org, p + q * np.exp(r * x_org),
                label=f'Initial estimate {p:.4}, {q:.4}, {r:.4}')
        ax.legend()

    return tr.detransform(a, b, c)


def fit_single(t, v, plot=False):
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
            fig = plt.figure(figsize=(8, 9))
            ax0 = fig.add_subplot(2, 1, 1)
        fig.subplots_adjust(0.095, 0.06, 0.995, 0.995, wspace=0.3, hspace=0.4)
        ax0.set_xlabel('x')
        ax0.set_ylabel('y')
        code, color = ('-', '#92cc92') if len(x) > 10 else ('x-', 'tab:green')
        ax0.plot(x, y, code, color=color, label='Transformed data')
    else:
        ax0 = None

    # Get an initial estimate
    at0, bt0, ct0 = estimate_initial_single(x, y, axes=ax0, vet=False)

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
        fit_label = f'rmse {np.sqrt(r.score):.4}'
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
            ax0.text(0.75, -0.32, '\n'.join(lines), transform=ax0.transAxes,
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


def fit_single_tau(t, v):
    """
    Fits a single exponential and returns a time constant.
    """
    c = fit_single(t, v)[2]
    if c == 0:
        # Instead of checking sign of zero and returning + or - inf, let numpy
        # handle it (but silently)
        with np.errstate(divide='ignore'):
            return -1 / c
    return -1 / c

