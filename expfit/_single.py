#
# Single expontial fits, form the basis of multi-exponential fits
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import numpy as np

import expfit


def estimate_initial_single(x, y, use_fallbacks=True, axes=None, vet=True):
    """
    Estimate ``a, b, c`` in ``y = a + b * exp(c * x)`` using derivatives
    estimated from mean averages at the sides.

    The method first uses :meth:`find_linear_segment` to find two data
    segments, one at the start of the data and one near the end, which are
    well approximated by a straight line. From these segments it then derives
    ``(x1, y1, dydx1)`` and ``(x2, y2, dydx2)`` (where the derivative is
    approximated by the slope of the straight line). It then estimates c from

        y    = a + b * exp(c * x)
        dydx = c * b * exp(c * x)

        y_1    - y_2    =     b * (exp(c * x_1) - exp(c * x_2))
        dydx_1 - dydx_2 = c * b * (exp(c * x_1) - exp(c * x_2))
        c = (dydx_1 - dydx_2) / (y_1 - y_2)

    Either segment can then be used to derive ``a`` and ``b``, from

        a = y_i - dydx_i / c
        b = (y_i - a) / np.exp(c * x_i)

    To pick a segment, the method assumes that the side with the steepest slope
    will have the best signal to noise ratio.

    Arguments:

    ``x``
        A time vector
    ``y``
        The corresponding values
    ``use_fallbacks=True``
        Set this to ``True`` (default, and recommended) to fall back on coarser
        estimation if the segments found by `find_linear_segment` return slopes
        that cannot correspond to an exponential.
    ``axes=None``
        Pass in matplotlib Axes to obtain a plot of the selected segments and
        estimates slopes.
    ``vet=True``
        This can be used to disable checks on the dimensions of ``x`` and'
        ``y``.

    If a matplot ``axes`` object is passed in, it will plot the used line
    segments.

    Returns a tuple ``(a, b, c)``.
    """
    if vet:
        x, y = expfit.vet_series(x, y)
    n = len(x)
    if n < 3:
        raise ValueError('At least 3 points are required')

    # Obtain segments to base estimates on
    use_fallback_s0 = use_fallback_s1 = False
    m_min = 5
    m_max = (n + 1) // 2
    _, s_all = expfit.least_squares(x, y, vet=False)
    if n > 2 * m_min:
        # Estimate the slope of the left-most and right-most segments of the
        # data. Use a segment that provides a good linear fit.
        if n >= 1000:
            m = max(m_min, n // 10)
        elif n >= 100:
            m = max(m_min, n // 3)
        else:
            m = max(m_min, n // 2)

        # Find segments
        xlo, ylo, olo, s0 = expfit.find_linear_segment(
            x[:m], y[:m], m_min, vet=False)
        xhi, yhi, ohi, s1 = expfit.find_linear_segment(
            x[-m:], y[-m:], m_min, left=False, vet=False)

        # Fall-back routines
        if use_fallbacks:
            # Check 1: Segment slopes should have the same sign as the slope of
            #          the full signal
            if s0 * s_all < 0:
                # print('Fall back sign s0')
                use_fallback_s0 = True
            if s1 * s_all < 0:
                # print('Fall back sign s1')
                use_fallback_s1 = True

            # Check 2: One slope must have a larger, one a smaller magnitude
            #          than the full signal
            mag, mag0, mag1 = abs(s_all), abs(s0), abs(s1)
            if ((mag0 > mag and mag1 > mag) or (mag0 < mag and mag1 < mag)):
                # Benefit of the doubt: retain largest mag
                if mag0 >= mag1:
                    # print('Fall back mag s1')
                    use_fallback_s1 = True
                else:
                    # print('Fall back mag s0')
                    use_fallback_s0 = True
    else:
        use_fallback_s0 = use_fallback_s1 = True

    # Not enough points or clever method failed: use half of signal
    if use_fallback_s0:
        xlo, ylo = x[:m_max], y[:m_max]
        x0, y0 = np.mean(xlo), np.mean(ylo)
        olo, s0 = expfit.least_squares(xlo, ylo, vet=False)
        if s0 * s_all < 0:
            olo, s0 = y0, 0.0
    if use_fallback_s1:
        xhi, yhi = x[-m_max:], y[-m_max:]
        x1, y1 = np.mean(xhi), np.mean(yhi)
        ohi, s1 = expfit.least_squares(xhi, yhi, vet=False)
        if s1 * s_all < 0:
            ohi, s1 = y1, 0.0

    # Use means of selected segments
    x0, x1, y0, y1 = np.mean(xlo), np.mean(xhi), np.mean(ylo), np.mean(yhi)

    # Show segments in plot
    if axes is not None:  # pragma: no cover
        axes.plot(x0, y0, 'ks', zorder=10)
        axes.plot(xlo, olo + s0 * xlo, 'k', zorder=10,
                  label=f'Slope {s0:.3}, n={len(xlo)}')
        axes.plot(x1, y1, 'rs', zorder=10)
        axes.plot(xhi, ohi + s1 * xhi, 'r', zorder=10,
                  label=f'Slope {s1:.3}, n={len(xhi)}')

    # Avoid divide by zero if y0 == y1: straight line
    if y0 - y1 == 0:
        return np.mean(y), 0.0, 0.0  # Actually use .0 here!

    # Estimate c, a, and b
    c = (s0 - s1) / (y0 - y1)
    if c < 0 or s1 == 0:
        a = y0 - (s0 / c if s0 != 0 else 0)
        b = (y0 - a) * np.exp(-c * x0)
    else:
        a = y1 - (s1 / c if s1 != 0 else 0)
        b = (y1 - a) * np.exp(-c * x1)

    # Flat line a better fit?
    with np.errstate(over='ignore', divide='ignore'):
        rmse_expf = np.sqrt(np.sum((y - a - b * np.exp(c * x))**2) / len(x))
        rmse_flat = np.sqrt(np.sum(np.sum((y - a)**2)) / len(x))
        rmse_ratio = rmse_expf / rmse_flat
    if use_fallbacks and rmse_ratio > 2:
        # Straight line is a much better fit
        return np.mean(y), 0.0, 0.0

    return a, b, c


class SingleExponentialError():
    """
    Callable class returning the MSE and its Jacobian and Hessian for a single
    exponential ``y = a + b * exp(c * x)`` fit with parameter set
    ``p = (a, b, c)``.
    """
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._m = 1 / len(x)

    def __call__(self, p):
        a, b, c = p
        e = np.exp(c * self._x)
        f = a - self._y + b * e
        ef = e * f
        mse = self._m * np.sum(f * f)

        # Jacobian
        jac = np.array([
            2 * self._m * np.sum(f),
            2 * self._m * np.sum(ef),
            2 * self._m * np.sum(ef * self._x) * b
        ])

        # Hessian
        ex = e * self._x
        aex = (a - self._y + 2 * b * e) * ex
        hes = np.array([
            [2, 2 * self._m * np.sum(e), 2 * b * self._m * np.sum(ex)],
            [0, 2 * self._m * np.sum(e * e), 2 * self._m * np.sum(aex)],
            [0, 0, 2 * self._m * b * np.sum(self._x * aex)],
        ])
        hes[1, 0] = hes[0, 1]
        hes[2, 0] = hes[0, 2]
        hes[2, 1] = hes[1, 2]

        return mse, jac, hes


def fit_single(t, v, plot=False):
    """
    Fits an exponential ``a + b * exp(c * t)`` to the time series ``(t, v)``,
    returning ``(a, b, c)``

    Example::

        t = np.linspace(0, 1, 100)
        v = 3 - 2 * np.exp(4 * t) + np.random.normal(0, 1)
        a, b, c = expfit.fit_single(t, v)
        print(a, b, c)

    """
    t, v = expfit.vet_series(t, v)

    # Transform to unit square, to avoid overflows
    rt, rv = (t[-1] - t[0]), (v[-1] - v[0])
    if rv == 0:
        rv = 1
    x, y = (t - t[0]) / rt, (v - v[0]) / rv

    # Create initial plot
    known = False
    try:
        if len(plot) == 3:
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
    e = SingleExponentialError(x, y)
    with np.errstate(all='ignore'):
        r = expfit.fmin(e, (at0, bt0, ct0))
    at, bt, ct = r.x

    # Detransform obtained parameters
    a = v[0] + at * rv
    b = bt * rv * np.exp(-ct * t[0] / rt)
    c = ct / rt

    if plot:  # pragma: no cover
        a0 = v[0] + at0 * rv
        b0 = bt0 * rv * np.exp(-ct0 * t[0] / rt)
        c0 = ct0 / rt

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

