#
# Initial estimates of single exponential fits.
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import numpy as np

import expfit


class SingleExponentialEstimate:
    """
    Estimated parameters of a single exponential ``a + b * exp(-x / tau)``.

    Can be used as a (read-only) sequence, or may provide extra information if
    :meth:`estimate_initial_single` is called with ``full=True``. In this case
    the following extra properties will be set:

    ``log1``
        A list where each entry is ``(least_squares_fit, message)``
        containing a least squares fit to a segment at the start of the signal,
        and a message describing it.
    ``log2``
        Like ``log1``, but for the end of the signal.
    ``region``
        Either ``None``, or the lower and upper indices of the region zoomed in
        on.

    """
    def __init__(self, a, b, tau):
        self._p = np.array([a, b, tau], dtype=float)
        self.log1 = None
        self.log2 = None
        self.region = None

    def __len__(self):
        return 3

    def __getitem__(self, subscript):
        return self._p.__getitem__(subscript)

    def __str__(self):
        return ' '.join(f'{i:.4g}' for i in self._p)


def estimate_initial_single(tv, full=False, plot=False):
    """
    Estimate ``a, b, tau`` in ``y = a + b * exp(-x / tau)`` using derivatives
    estimated from mean averages at the sides.

    The method first selects two segments, one at the start of the signal and
    one near the end, and approximates them with a straight line to derive
    ``(x1, y1, dydx1)`` and ``(x2, y2, dydx2)``. It then estimates c from

        y    = a + b * exp(-x / tau)
        dydx = c * b * exp(-x / tau)

        y_1    - y_2    = b * (exp(c * y_1) - exp(c * y_2))
        dydx_1 - dydx_2 = b * (exp(c * y_1) - exp(c * y_2)) / -tau
        tau = (y_2 - y_1) / (dydx_1 - dydx_2)

    Either segment can then be used to derive ``a`` and ``b``, from

        a = y_i + dydx_i tau
        b = (y_i - a) / exp(-x_i / tau)

    To pick a segment, the method starts by splitting the series down the
    middle, and performing a linear least squares fit on each half. If this
    contains an exponential, both slopes should have the same sign, but a
    different magnitude. The segments are then refined by successive halving,
    and accepted as a better segment if the slope in the steep part gets
    steeper, or if the slope in the shallow part gets shallower.

    If the time series does not appear to contain an exponential, a
    :class:`NotExponentialError` is returned.

    Example::

        t = np.linspace(0, 1, 50)
        v = 1 + 3 * np.exp(-0.5 * t)
        tr = expfit.UnitSquareTransform(t, v)
        q = expfit.estimate_initial_single(tr.x, tr.y)
        a, b, tau = tr.detransform(q)
        print(a, b, tau)

    Arguments:

    ``tv``
        A tuple ``(t, v)`` containing a time series.
    ``full=False``
        Set to ``True`` to store debugging and visualisation information in
        the returned :class:`SingleExponentialEstimate`.
    ``plot=False``
        Set to ``True`` to create a plot of the initial estimation process.
        Setting this to ``True`` will has the side effect of setting
        ``full=True``.

    Returns a :class:`SingleExponentialEstimate` with the estimated
    ``(a, b, tau)``.
    """
    if not isinstance(tv, expfit.TimeSeries):
        tv = expfit.TimeSeries(*tv)
    x_nozoom, y_nozoom = tv
    if len(x_nozoom) < 3:
        raise ValueError('At least 3 points are required')

    # Full information is returned if plot=True
    full = full or plot

    # Select a subsection of the data, if the signal is too steep
    zoom_region = find_action(x_nozoom, y_nozoom)
    if zoom_region is None:
        x, y = x_nozoom, y_nozoom
    else:
        i, j = zoom_region
        x, y = x_nozoom[i:j], y_nozoom[i:j]

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
    def shrink(seg, ls, n_min, start=True, increasing=True, log=None):
        increasing = bool(increasing)
        x, y = seg

        n = len(x)
        if n < n_min:
            if log is not None:
                log.append((ls, 'Initial segment at minimum size'))
            return seg, ls

        r = None
        msg = None
        l_new = ls
        for i in range(n):  # Avoid endless loop
            # Propose new segment
            n = (1 + n) // 2
            x_new, y_new = (x[:n], y[:n]) if start else (x[-n:], y[-n:])
            l_new = expfit.LeastSquaresFit(x_new, y_new, vet=False)

            # Stop if too small
            if n < n_min:
                msg = f'Minimum size reached ({n} < {n_min})'
                break

            # Test slope sign and magnitude
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
            if log is not None:
                log.append((ls, f'Slope {ls.slope:.3}, n={len(x)}'))

        # Sanity check: endless loop
        assert i - 1 < len(seg[0]), 'Maximum iterations unexpectedly reached'

        # Log last proposed segment
        if log is not None:
            log.append((l_new, msg))

        return (x, y), ls

    # Get starting segments, and least squares fits
    m = (1 + len(x)) // 2
    seg1 = x[:m], y[:m]
    seg2 = x[-m:], y[-m:]
    l0 = expfit.LeastSquaresFit(x, y, vet=False)
    l1 = expfit.LeastSquaresFit(*seg1, vet=False)
    l2 = expfit.LeastSquaresFit(*seg2, vet=False)

    # Log shrinking process for plot?
    log1, log2 = ([], []) if full else (None, None)

    # Slopes must match full signal slope (otherwise this is either slow drift
    # or correlated noise at the flat end of the exponential, or the signal is
    # not an exponential).
    # Slopes ok? Then start shrinking
    n_min = 5
    if l0.slope * l1.slope < 0:
        l1.slope, l1.offset = 0.0, l1.mu_y
        if log1 is not None:
            log1.append((l1, 'Slope set to zero'))
    else:
        seg1, l1 = shrink(
            seg1, l1, n_min, True, abs(l1.slope) > abs(l2.slope), log1)

    if l0.slope * l2.slope < 0:
        l2.slope, l2.offset = 0.0, l2.mu_y
        if log2 is not None:
            log2.append((l2, 'Slope set to zero'))
    else:
        seg2, l2 = shrink(
            seg2, l2, n_min, False, abs(l2.slope) > abs(l1.slope), log2)

    x1, y1, s1 = l1.mu_x, l1.mu_y, l1.slope
    x2, y2, s2 = l2.mu_x, l2.mu_y, l2.slope

    # Edge cases
    if s1 == s2:
        raise expfit.NotExponentialError('Equal slopes')
    elif y1 == y2:
        raise expfit.NotExponentialError('Equal means')

    # Estimate tau and both (a, b) sets
    tau = (y2 - y1) / (s1 - s2)
    a1 = y1 + s1 * tau
    a2 = y2 + s2 * tau
    b1 = (y1 - a1) * np.exp(x1 / tau)
    b2 = (y2 - a2) * np.exp(x2 / tau)
    am, bm = np.mean((a1, a2)), np.mean((b1, b2))

    # Use start, end, or averaged parameters, depending on RMSE
    with np.errstate(over='ignore', divide='ignore'):
        r1 = expfit.rmse(x, y, (a1, b1, tau))
        r2 = expfit.rmse(x, y, (a2, b2, tau))
        rm = expfit.rmse(x, y, (am, bm, tau))
        rs = expfit.rmse(x, y, (l0.mu_y, 0, 1))
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
        if (r / rs) > 1:
            raise expfit.NotExponentialError('Flat line is better fit')

    # Create results object
    r = SingleExponentialEstimate(a, b, tau)
    if full:
        r.log1 = log1
        r.log2 = log2
        r.region = zoom_region

    # Show initial estimate
    if plot:  # pragma: no cover
        from ._plot import initial_estimate_plot
        initial_estimate_plot(x_nozoom, y_nozoom, r)

    return r


def find_action(x, y, r_factor=20, n_min=10):
    """
    For very steep exponentials, isolates a region of the series ``(x, y)`` for
    use in initial estimates.

    The method tests wether there is a segment at the start or end of the
    signal, in which the range of ``y`` exceeds ``r_factor`` times the range
    outside this segment. If this segment exists, and has length greather than
    ``n_min``, the method returns the indices corresponding to that segment. If
    no such segment is found, ``None`` is returned.

    Example::

        zoom_region = find_action(x, y)
        if zoom_region is not None:
            i, j = zoom_region
            x, y = x[i:j], y[i:j]

    """
    n = len(y)
    m = n // 2
    s1, s2 = y[:m], y[m:]
    r1, r2 = np.max(s1) - np.min(s1), np.max(s2) - np.min(s2)

    if r2 != 0 and r1 / r2 > r_factor:
        while r2 != 0 and r1 / r2 > r_factor and m > 1:
            m = max(m // 2, 1)
            s1, s2 = y[:m], y[m:]
            r1, r2 = np.max(s1) - np.min(s1), np.max(s2) - np.min(s2)

        if m >= n_min:
            return 0, m

    elif r1 != 0 and r2 / r1 > r_factor:
        while r1 != 0 and r2 / r1 > r_factor and m > 1:
            m = max(m // 2, 1)
            s1, s2 = y[:-m], y[-m:]
            r1, r2 = np.max(s1) - np.min(s1), np.max(s2) - np.min(s2)

        if m >= n_min:
            return n - m, n

    return None


def estimate_initial_opposing(x, y, plot=False, vet=True):
    """


    TODO



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
    imn, imx = 3 + np.argmin(y[3:-3]), 3 + np.argmax(y[3:-3])
    mn = max(abs(y[0] - y[imn]), abs(y[-1] - y[imn]))
    mx = max(abs(y[0] - y[imx]), abs(y[-1] - y[imx]))
    isplit = imn if mn > mx else imx

    # Fit exponentials to both segments
    p0 = a0, b0, c0 = expfit.estimate_initial_single(
        x[isplit:], y[isplit:], vet=False)
    a1, b1, c1 = expfit.estimate_initial_single(
        x[:isplit], y[:isplit] - expfit.expc(x[:isplit], p0), vet=False)

    # Create plot
    if plot:  # pragma: no cover

        # TODO MOVE

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot()
        ax.plot(x, y, 's-' if len(x) < 50 else '-', label='Data')
        ax.axvline(x[isplit], color='k', ls='--', lw=1, label='Split')
        ax.plot(x, expfit.expc(x, p0), 'r', label='Dominant')
        ax.plot(x[:isplit], a1 + y[:isplit] - expfit.expc(x[:isplit], p0),
                label='Data with dominant subtracted')
        ax.plot(x, expfit.expc(x, (a0, b1, c1)), 'k', label='Second')
        ax.plot(x, expfit.expc(x, (a0, b0, c0, b1, c1)), label='Combined')
        ax.legend()

    return a0, b0, c0, b1, c1


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
    sigma = np.std(r)

    if plot:  # pragma: no cover
        from ._plot import sigma_plot
        sigma_plot(x, y, xx, yy, r, sigma)

    return sigma

