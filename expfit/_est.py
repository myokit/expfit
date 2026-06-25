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
    Estimated parameters of a single exponential ``a + b * exp(c * x)``.

    Can be used as a (read-only) sequence, or may provide extra information if
    :meth:`estimate_initial_single` is called with ``full=True``. In this case
    the following extra properties will be set:

    ``ls1``
        A linear least squares fit to the selected segment at the start of the
        signal.
    ``ls2``
        A linear least squares fit to the selected segment at the end of the
        signal.
    ``log1``
        A list where each entry is ``(least_squares_fit, message)``
        containing a proposed least squares fit to a segment at the start of
        the signal, and a message describing it.
    ``log2``
        Like ``log1``, but for the end of the signal.
    ``region``
        Either ``None``, or the lower and upper indices of the region zoomed in
        on.

    """
    def __init__(self, a, b, c):
        self._p = np.array([a, b, c], dtype=float)
        self.ls1 = None
        self.ls2 = None
        self.log1 = None
        self.log2 = None
        self.region = None

    def __len__(self):
        return 3

    def __getitem__(self, subscript):
        return self._p.__getitem__(subscript)

    def __str__(self):
        return ' '.join(f'{i:.4g}' for i in self._p)


def estimate_initial_single(x, y=None, full=False, plot=False):
    """
    Estimates ``a, b, c`` in ``y = a + b * exp(c * x)`` using derivatives
    estimated from mean averages at the sides.

    The method first selects two segments, one at the start of the signal and
    one near the end, and approximates them with a straight line to derive
    ``(x1, y1, dydx1)`` and ``(x2, y2, dydx2)``. It then estimates c from

        y    = a + b * exp(c * x)
        dydx = c * b * exp(c * x)

        y_1    - y_2    = b * (exp(c * y_1) - exp(c * y_2))
        dydx_1 - dydx_2 = b * (exp(c * y_1) - exp(c * y_2)) * c
        c = (dydx_1 - dydx_2) / (y_1 - y_2)
        b = (y1 - y2) / (exp(c * x1) - exp(c * x2)
        a = y_1 + dydx_1 / c

    To pick a segment, the method starts by splitting the series down the
    middle, and performing a linear least squares fit on each half. If this
    contains an exponential, both slopes should have the same sign, but a
    different magnitude.

    If this condition is met, the slopes are then refined by successive
    halving, with each halving accepted if:
      - The new segment contains at least 2 points
      - The slope of a linear fit to the new segment has the same sign as the
        slope of the previous segment
      - The area under the estimated exponential is more similar to the area
        under the data than the previous segment

    #   - the new slope is increased, on the fast side, or decreased, on the
    #     slow side

    If the time series does not appear to contain an exponential, a
    :class:`NotExponentialError` is raised.

    Example::

        x = np.linspace(0, 1, 50)
        y = 1 + 3 * np.exp(2 * x)
        t = expfit.UnitSquaredSeries(x, y)
        q = expfit.estimate_initial_single(t)
        a, b, c = tr.detransform(q)
        print(a, b, c)

    Arguments:

    ``x``, ``y``
        The time series as two one-dimensional arrays of equal size.
        Alternatively, ``x, y`` can be a :class:`TimeSeries` and ``None``.
    ``full=False``
        Set to ``True`` to store debugging and visualisation information in
        the returned :class:`SingleExponentialEstimate`.
    ``plot=False``
        Set to ``True`` to create a plot of the initial estimation process.
        Setting this to ``True`` will has the side effect of setting
        ``full=True``.

    Returns a :class:`SingleExponentialEstimate` with the estimated
    ``(a, b, c)``.
    """
    x_nozoom, y_nozoom = expfit.TimeSeries._from_xy(x, y)
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

    # Get starting segments, and least squares fits
    m = (1 + len(x)) // 2
    seg1 = x[:m], y[:m]
    seg2 = x[-m:], y[-m:]
    l0 = expfit.LeastSquaresFit(x, y)
    l1 = expfit.LeastSquaresFit(*seg1)
    l2 = expfit.LeastSquaresFit(*seg2)

    # Slopes must match full signal slope (otherwise this is either slow drift
    # or correlated noise at the flat end of the exponential, or the signal is
    # not an exponential).
    # Slopes ok? Then start shrinking
    shrink1 = shrink2 = True
    if l0.slope * l1.slope < 0:
        l1.slope, l1.offset = 0.0, l1.mu_y
        shrink1 = False
    if l0.slope * l2.slope < 0:
        l2.slope, l2.offset = 0.0, l2.mu_y
        shrink2 = False
    if not (shrink1 or shrink2):
        raise expfit.NotExponentialError('Not a (single) exponential')

    # Store initial segments
    log1 = log2 = None
    if full:
        log1 = [l1]
        log2 = [l2]

    # Calculate area under the data
    A0 = expfit._trapezoid(y, x)

    # Calculate a, b, c, and area
    def abca(l1, l2):
        x1, y1, s1 = l1.mu_x, l1.mu_y, l1.slope
        x2, y2, s2 = l2.mu_x, l2.mu_y, l2.slope
        if s1 == s2:
            return 0, 0, 0, 0

        c = (s1 - s2) / (y1 - y2)
        b = (y1 - y2) / (np.exp(c * x1) - np.exp(c * x2))
        a = y1 - s1 / c
        A = b / c * (np.exp(c * x[-1]) - np.exp(c * x[0])) + a * (x[-1] - x[0])
        return a, b, c, A

    # Shrink segments
    n_min = 2
    a, b, c, A = abca(l1, l2)

    shrunk1 = shrunk2 = True
    while shrunk1 or shrunk2:
        shrunk1 = False
        if shrink1 and l1.n > n_min:
            n = max(n_min, (1 + l1.n) // 2)
            sn = (seg1[0][:n], seg1[1][:n])
            ln = expfit.LeastSquaresFit(*sn)
            if ln.slope * l1.slope > 0:
                an, bn, cn, An = abca(ln, l2)
                if abs(An - A0) < abs(A - A0):
                    seg1, l1, a, b, c, A = sn, ln, an, bn, cn, An
                    shrunk1 = True
                    if log1 is not None:
                        log1.append(l1)

        shrunk2 = False
        if shrink2 and l2.n > n_min:
            n = max(n_min, (1 + l2.n) // 2)
            sn = (seg2[0][-n:], seg2[1][-n:])
            ln = expfit.LeastSquaresFit(*sn)
            if ln.slope * l2.slope > 0:
                an, bn, cn, An = abca(l1, ln)
                if abs(An - A0) < abs(A - A0):
                    seg2, l2, a, b, c, A = sn, ln, an, bn, cn, An
                    shrunk2 = True
                    if log2 is not None:
                        log2.append(l2)

    # Edge cases
    if l1.slope == l2.slope:
        raise expfit.NotExponentialError('Equal slopes')
    elif l1.mu_y == l2.mu_y:
        raise expfit.NotExponentialError('Equal means')

    # Create results object
    r = SingleExponentialEstimate(a, b, c)
    if full:
        r.ls1 = l1
        r.ls2 = l2
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

