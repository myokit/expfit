#
# Initial estimates of single exponential fits.
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import numpy as np

import expfit


class LeastSquaresFit():
    """
    Creates a least squares fit ``(offset, slope)`` where ``y`` is approximated
    by ``offset + slope * x``.

    Arguments:

    ``x``, ``y``
        Two equal-sized, 1d arrays.

    Public properties:

    ``offset``, ``slope``
        The fit, using ``y = offset + slope * x``
    ``mu_x``, ``mu_y``
        The mean ``x`` and ``y`` on the segment fit to.
    ``x``, ``y``
        Arrays containing the minimum and maximum ``x`` fit too, and the
        corresponding ``y`` values. Useful for plotting the straight line fit.
    ``n``
        The number of points in the original ``x`` (and in ``y``).

    """
    def __init__(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError('Both arrays must be 1-dimensional.')
        n = len(x)
        if n != len(y):
            raise ValueError('Both arrays must have same length.')
        if n < 2:
            raise ValueError('At least 2 points are required')

        self.mu_x = np.mean(x)
        self.mu_y = np.mean(y)
        xx = np.sum(x**2) - n * self.mu_x**2
        xy = np.sum(x * y) - n * self.mu_x * self.mu_y
        self.slope = xy / xx
        self.offset = self.mu_y - self.slope * self.mu_x
        self.x = np.array((x[0], x[-1]), dtype=float)
        self.y = self.offset + self.slope * self.x
        self.n = n

    def __repr__(self):
        return f'<expfit.LeastSquaresFit({self.offset:.3}+{self.slope:.3}x)>'

    def __str__(self):
        return (f'mu ({self.mu_x:.3}, {self.mu_y:.3}),'
                f' {self.offset:.3} + {self.slope:.3} x')


class SingleExponentialEstimate:
    """
    Estimated parameters of a single exponential ``a + b * exp(c * x)``.

    Can be used as a (read-only) sequence, or may provide extra information if
    :meth:`est1` is called with ``full=True``. In this case the following
    additional properties will be set:

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


def est1(x, y=None, reject_linear=True, full=False, plot=False):
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

    If the time series does not appear to contain an exponential, a
    :class:`NotExponentialError` is raised.

    Example::

        x = np.linspace(0, 1, 50)
        y = 1 + 3 * np.exp(2 * x)
        t = expfit.UnitSquaredSeries(x, y)
        q = expfit.est1(t)
        a, b, c = tr.detransform(q)
        print(a, b, c)

    Arguments:

    ``x``, ``y``
        The time series as two one-dimensional arrays of equal size.
        Alternatively, ``x, y`` can be a :class:`TimeSeries` and ``None``.
    ``reject_linear=True``
        By default, the result is compared to a linear least squares fit and
        rejected if the straight line fit is comparable. This parameter can be
        used to disable this check.
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
    xy_no_zoom = expfit.TimeSeries._from_xy(x, y)
    if len(xy_no_zoom[0]) < 3:
        raise ValueError('At least 3 points are required')
    del x, y

    # Full information is returned if plot=True
    full = full or plot

    # Select a subsection of the data, if the signal is too steep
    zoom_region = find_action(xy_no_zoom)
    x, y = xy_no_zoom
    if zoom_region is not None:
        i, j = zoom_region
        x, y = x[i:j], y[i:j]

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
        with np.errstate(all='raise', divide='raise'):
            try:
                c = (s1 - s2) / (y1 - y2)
                e1, e2 = np.exp(c * x1), np.exp(c * x2)
                b = (y1 - y2) / (e1 - e2)
                a = y1 - b * e1
                e1, e2 = np.exp(c * x[-1]), np.exp(c * x[0])
                A = b / c * (e1 - e2) + a * (x[-1] - x[0])
            except FloatingPointError:
                return 0, 0, 0, 0
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

    # Create results object
    r = SingleExponentialEstimate(a, b, c)
    if full:
        r.ls1 = l1
        r.ls2 = l2
        r.log1 = log1
        r.log2 = log2
        r.region = zoom_region

    # Show initial estimate (before failing)
    if plot:  # pragma: no cover
        from ._plot import initial_estimate_plot
        initial_estimate_plot(xy_no_zoom[0], xy_no_zoom[1], r)

    # Catch silent failures in abca()
    if l1.slope == l2.slope:
        raise expfit.NotExponentialError('Equal slopes')
    elif l1.mu_y == l2.mu_y:
        raise expfit.NotExponentialError('Equal means')

    # Catch less obviously straight lines
    if reject_linear:
        n = len(x)
        x = (y - a - b * np.exp(c * x))
        m1 = np.sum((y - a - b * np.exp(c * x))**2) / n
        m2 = np.sum((y - l0.offset - l0.slope * x)**2) / n
        # Akaike cut-off
        line = (m2 <= m1 * (2 + n) / n)
        if not line and m1 == 0:
            # Ad-hoc comparison for m1 == 0, m2 almost 0
            line = m2 / abs(A0) < 1e-9
        if line:
            raise expfit.NotExponentialError('Straight line is better fit')

    return r


def find_action(x, y=None, r_factor=20, n_min=10):
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

    Arguments:

    ``x``, ``y``
        The time series as two one-dimensional arrays of equal size.
        Alternatively, ``x, y`` can be a :class:`TimeSeries` and ``None``.
    ``r_factor``
        Ratio between ranges that triggers zooming in.
    ``n_min``
        Minimum size of zoomed-in on segment.

    Returns the lower and upper indices of the segment where the action
    happens, or ``None`` if no stand-out segment is found.
    """
    x, y = expfit.TimeSeries._from_xy(x, y)
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


def estd11(x, y=None, sigma=None, plot=False):
    """
    Estimates parameters for two decaying exponentials with opposite signs.

    Arguments:

    ``x``, ``y``
        The time series as two one-dimensional arrays of equal size.
        Alternatively, ``x, y`` can be a :class:`TimeSeries` and ``None``.
    ``sigma``
        An estimate of the noise level in the data, given as a standard
        deviation.
    ``plot=False``
        Set to ``True`` to create a debugging plot.

    Returns a parameter set ``(a, b0, c0, b1, c1)`` where ``b0`` and ``c0`` are
    for the slower exponential in the second part of the signal.
    """
    x, y = xy = expfit.TimeSeries._from_xy(x, y)
    if len(x) < 10:
        raise ValueError('At least 10 points are required')

    # Estimate start, end, max, and min
    # Skip points, to ensure at least a segment of length n_min
    n_min = 3
    imn = n_min + np.argmin(y[n_min:-n_min])
    imx = n_min + np.argmax(y[n_min:-n_min])
    mn = max(abs(y[0] - y[imn]), abs(y[-1] - y[imn]))
    mx = max(abs(y[0] - y[imx]), abs(y[-1] - y[imx]))
    isplit = imn if mn > mx else imx

    # Fit exponential to second (dominant) segment
    p0 = p1 = msg = None
    x0, y0 = x[isplit:], y[isplit:]
    try:
        a0, b0, c0 = p0 = expfit.est1(x0, y0, reject_linear=False)

        # Subtract fit-to-second from first
        with np.errstate(over='raise'):
            x1, y1 = x[:isplit], y[:isplit] - expfit.exp1(x[:isplit], p0)
    except (FloatingPointError, expfit.NotExponentialError):
        msg = 'Second segment is not exponential'

    # Fit exponential to subtracted signal
    if p0 is not None:
        try:
            a1, b1, c1 = p1 = expfit.est1(x1, y1, reject_linear=False)
        except FloatingPointError:
            msg = 'Second segment is not exponential'
        except expfit.NotExponentialError:
            msg = 'First segment is not exponential'

    # Check results
    if p0 is not None and p1 is not None:
        if c0 > 0 or c1 > 0:
            msg = 'Segments not both decaying'
        elif b0 * b1 >= 0:  # pragma: no cover
            msg = 'Segment are exponential but do not have opposing signs'
        else:
            print(a0, b0, c0)
            print(a1, b1, c1)


            # Compare areas (without a)
            A0 = b0 / c0 * (np.exp(c0 * x[-1]) - np.exp(c0 * x[0]))
            A1 = b1 / c1 * (np.exp(c1 * x[-1]) - np.exp(c1 * x[0]))
            print(A0, A1)
            print()

            if abs(A1 / A0) < 1e-2:
                msg = ('Possible fit, but area under segments suggests highly'
                       ' unequal contributions.')

            '''
            # Catch less obviously straight lines
            if reject_linear:
                n = len(x)
                x = (y - a - b * np.exp(c * x))
                m1 = np.sum((y - a - b * np.exp(c * x))**2) / n
                m2 = np.sum((y - l0.offset - l0.slope * x)**2) / n
                # Akaike cut-off
                line = (m2 <= m1 * (2 + n) / n)
                if not line and m1 == 0:
                    # Ad-hoc comparison for m1 == 0, m2 almost 0
                    line = m2 / abs(A0) < 1e-9
                if line:
                    raise expfit.NotExponentialError('Straight line is better fit')
            '''


    # Create plot
    if plot:  # pragma: no cover
        from ._plot import initial_opposing_plot
        initial_opposing_plot(xy, isplit, p0, p1)

    # Raise exception or return
    if msg is not None:
        raise expfit.NotOpposingError(msg)

    return a0, b0, c0, b1, c1


def estimate_noise_level(x, y=None, plot=False):
    """
    Estimates the noise level in a signal --- assuming it is well fit by either
    a straight line or a single exponential.

    A typical use case would be to run this on a known flat bit of signal, or
    on the final 10% of a decaying exponential signal.

    Arguments:

    ``x``, ``y``
        The time series as two one-dimensional arrays of equal size.
        Alternatively, ``x, y`` can be a :class:`TimeSeries` and ``None``.
    ``plot``
        Optional parameter to create a plot of the method's workings.

    Returns ``sigma`` where ``sigma**2`` is the variance of a normal
    distribution with the estimated noise level.
    """
    x, y = xy = expfit.TimeSeries._from_xy(x, y)
    if len(x) < 10:
        raise ValueError('At least 10 points are required')







    xx, yy = x[-m:], y[-m:]
    try:
        p0 = expfit.fit1(xx, yy)
    except expfit.NotExponentialError:
        # Very steep? Then use all data for fit (but not for residuals)
        p0 = expfit.fit1(x, y)
    r = yy - expfit.exp1(xx, p0)
    sigma = np.std(r)

    if plot:  # pragma: no cover
        from ._plot import sigma_plot
        sigma_plot(x, y, xx, yy, r, sigma)

    return sigma

