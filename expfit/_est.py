#
# Initial estimates of single exponential fits.
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

    To pick a segment, the method starts by splitting the series down the
    middle, and performing a linear least squares fit on each half. If this
    contains an exponential, both slopes should have the same sign, but a
    different magnitude. The segments are then refined by successive halving,
    and accepted as a better segment if the slope in the steep part gets
    steeper, or if the slope in the shallow part gets shallower.

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
        def plot_line(ax, x, ls, start=True, msg=None, dotted=False):
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

            line = ':' if dotted else '-'
            fill = 'none' if dotted else 'full'
            ax.plot(x, y, color=color, ls=line, zorder=10, label=label)
            ax.plot(mu_x, mu_y, 's', color=color, fillstyle=fill, zorder=10)

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
                msg = 'Slope sign changed'
                break
            if (abs(l_new.slope) >= abs(ls.slope)) != increasing:
                msg = 'Slope decreased' if increasing else 'Slope increased'
                break

            # Test near-constant rate of change
            r_new = l_new.slope / ls.slope
            if r is not None:
                rr = r / r_new
                if rr < 0.8 or rr > 1.25:
                    msg = f'Unexpected slope change ratio {rr:.3}'
                    break

            # Accept
            r = r_new
            x, y, ls = x_new, y_new, l_new
            if n == n_min:
                msg = 'Minimum size reached'

        # Show last segment
        if plot and not (x is x_new):  # pragma: no cover
            plot_line(ax, x_new, l_new, start, msg=msg, dotted=True)

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
    # Use the derived segments to estimate the parameters
    #

    # Return a consistent result when things go wrong
    def fail(seg1, seg2, l1, l2, msg):
        if plot:  # pragma: no cover
            plot_line(ax, seg1[0], l1, True, msg)
            plot_line(ax, seg2[0], l2, False, msg)
            ax.legend()
        return abc_fail

    # Unpack
    x1, y1, s1 = l1.mu_x, l1.mu_y, l1.slope
    x2, y2, s2 = l2.mu_x, l2.mu_y, l2.slope

    # Edge cases for c estimate
    if s1 == s2:
        return fail(seg1, seg2, l1, l2, 'Equal slopes (c=0)')
    if y1 == y2 or np.abs(y1 - y2) < 1e-16:
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
        r1 = expfit.rmse(x, y, (a1, b1, c))
        r2 = expfit.rmse(x, y, (a2, b2, c))
        rm = expfit.rmse(x, y, (am, bm, c))
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
        rr = r / expfit.rmse(x, y, (l0.mu_y, 0, 0))
    if rr > 2:
        return fail(seg1, seg2, l1, l2, 'Flat line is better fit')

    # Show initial estimate
    if plot:  # pragma: no cover
        p, q, r = tr.detransform(a, b, c)
        ax.plot(x_org, p + q * np.exp(r * x_org),
                label=f'Initial estimate {p:.4}, {q:.4}, {r:.4}')
        ax.legend()

    return tr.detransform(a, b, c)


def estimate_initial_opposing(x, y, plot=False, vet=True):
    """
    Split the time series ``(x, y)`` into two segments, trending in different
    directions.

    Arguments:

    ``x``, ``y``
        A time vector and the correspond values. Assumed to be transformed onto
        the unit square.
    ``plot=False``
        Set to ``True`` to create a debugging plot.
    ``vet=True``
        Set to ``False`` to disable checks on the dimensions of ``t`` and
        ``v``. This should only be done if the input data is already vetted.

    Returns .....
    """
    if vet:
        x, y = expfit.vet_series(x, y)
    if len(x) < 10:
        raise ValueError('At least 10 points are required')

    # Estimate start, end, max, and min
    # Skip points, to ensure at least a segment of length 3
    imn, imx = 3 + np.argmin(y[3:]), np.argmax(y[:-3])
    mn = max(abs(y[0] - y[imn]), abs(y[-1] - y[imn]))
    mx = max(abs(y[0] - y[imx]), abs(y[-1] - y[imx]))
    isplit = imn if mn > mx else imx

    # Fit exponentials to both segments
    p1 = expfit.estimate_initial_single(x[isplit:], y[isplit:], vet=False)
    p0 = expfit.estimate_initial_single(
        x[:isplit], y[:isplit] - expfit.exp(x[:isplit], p1), vet=False)
    a0, b0, c0 = p0
    a1, b1, c1 = p1

    # Create plot
    if plot:  # pragma: no cover
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot()
        ax.plot(x, y, 's-' if len(x) < 50 else '-')
        ax.axvline(x[isplit], color='tab:orange', lw=1)
        ax.plot(x[:isplit], a1 + y[:isplit] - expfit.exp(x[:isplit], p1))
        ax.plot(x, expfit.exp(x, (a1, b0, c0)), 'k')
        ax.plot(x, expfit.exp(x, p1), 'r')
        ax.plot(x, expfit.exp(x, (a1, b0, c0, b1, c1)), '--')

    return a1, b0, c0, b1, c1


def estimate_noise_level(x, y, vet=True, plot=False):
    """
    Estimates the noise level by subtracting a dominant exponential from the
    final section of the signal and assuming what remains is normally
    distributed noise.

    Arguments:

    ``x``, ``y``
        The series

    Returns ``sigma`` where ``sigma**2`` is the variance of a normal
    distribution with the estimated noise level.
    """
    if vet:
        x, y = expfit.vet_series(x, y)

    # Assume final part of signal is dominated by a single exponential
    n = len(x)
    m = min(max((n + 1) // 2, 10), n)

    #p0 = expfit.estimate_initial_single(x, y)
    xx, yy = x[-m:], y[-m:]
    p0 = expfit.fit1(xx, yy)
    r = yy - expfit.exp(xx, p0)

    if plot:  # pragma: no cover
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(14, 9))
        grid = fig.add_gridspec(3, 2)
        ax = fig.add_subplot(grid[0, 0])
        ax.plot(x, y)
        ax.plot(xx, yy)
        ax = fig.add_subplot(grid[1:, 0])
        ax.plot(xx, r)
        ax = fig.add_subplot(grid[1:, 1])
        ax.hist(r, bins='auto', density=True)
        var = np.std(r)**2
        hx = np.linspace(np.min(r), np.max(r), 99)
        hy = 1 / np.sqrt(2 * np.pi * var) * np.exp(-hx**2 / (2 * var))
        ax.plot(hx, hy)
        plt.show()

    return np.std(r)


def estimate_number_of_exponentials(x, y, p=0.9, vet=True):
    """
    Attempts to estimate the number of exponential components in a signal,
    without any fits.

    The method is based on the singular value decomposition of a Hankel matrix,
    and returns the number of singular values larger than a cut-off based on
    the set probability ``p``, according to

        p = e * sqrt(-2 n log(1 - p^(1 / n)))

    where ``e`` is an estimate of the signal noise variance from
    :meth:`estimate_noise_level`.

    This equation was suggested by Jeffrey M. Hokanson (2013) Numerically
    Stable and Statistically Efficient Algorithms for Large Scale Exponential
    Fitting. https://hdl.handle.net/1911/77161

    Returns an integer number close to the number of exponentials present, but
    sometimes over or underestimates by at least 1.
    """
    if vet:
        x, y = expfit.vet_series(x, y)

    # Construct Hankel matrix
    n = len(x)
    m = (n + 1) // 2
    h = np.zeros((m, m), dtype=float)
    for i in range(m):
        h[i] = y[i:i + m]

    # Calculate singular values (in descending order)
    s = np.linalg.svd(h, compute_uv=False)

    # Calculate cut-off, based on estimated noise level
    e = estimate_noise_level(x, y, vet=False)
    c = e * np.sqrt(-2 * n * np.log(1 - p**(1 / n)))
    return np.sum(s > c)

