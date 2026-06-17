#
# Single and multi-expontial fits
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import numpy as np

import expfit


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
        Optional parameter to create a plot of the method's workings. Can be
        set to ``True`` or to an array with the true ``(a, b, tau)``.

    Returns an :class:`ExponentialFit`.
    """
    t, v = expfit.vet_series(t, v)

    # Convert `plot` to boolean
    pt = plot
    plot = plot is not False

    # Transform to unit square, to avoid overflows etc
    tr = expfit.UnitSquareTransform(t, v)

    # Get an initial estimate (in transformed space)
    q0 = expfit.estimate_initial_single(tr.x, tr.y, full=plot, vet=False)

    # Stop if the signal is not exponential
    if q0[1] == 0 or q0[2] == 0:
        raise expfit.NotExponentialError()

    # Fit (in unit transformed space)
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
        from ._plot import fit1_plot
        try:
            assert len(pt) == 3
        except (TypeError, AssertionError):
            pt = None
        fit1_plot(t, v, tr, r, p, q0, pt)

    return p


def fitd2(t, v, plot=False, opt_plot=False):
    """
    Fits a decaying double-exponential to a time series, with equal signed
    multipliers for both components.

    Returns parameters for::

        v = a + b_0 * exp(-t / tau_0) + b_1 * exp(-t / tau_1)

    where ``tau_0 > tau_1``.

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

    # Convert `plot` to boolean
    pt = plot
    plot = plot is not False

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
    e = expfit.MultiExponentialError(t, v, 2, 0, b0 > 0)

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
            if plot:  # pragma: no cover
                print(r)
            opt_fig = r.plot
        if np.exp(r.x[4] - r.x[2]) > 1.1 and r.success:
            break
        elif i + 1 == max_iter:  # pragma: no cover
            raise RuntimeError(
                f'Unable to find good fit after {max_iter} attempts.')
    #print(f'Done in {1 + i} repeats. Last opt had {r.iterations} iter.')

    # Detransform parameters
    et = expfit.TauFormError(t, v)
    p = expfit.ExponentialFit(t, v, e.detransform(r.x, True), et)

    if plot:  # pragma: no cover
        from ._plot import tau_plot
        p0 = expfit.ExponentialFit(t, v, e.detransform(q0, True), et)
        pe = expfit.ExponentialFit(t, v, (a0, b0, -1 / c0), et)
        try:
            assert len(pt) == 5
        except (TypeError, AssertionError):
            pt = None
        tau_plot(t, v, r, p, p0, pe, pt)

    return p


def fitd11(t, v, plot=False, opt_plot=False):
    """
    Fits a decaying double-exponential to a time series, with opposite signed
    multipliers for both components.

    Returns parameters for::

        v = a + b_0 * exp(-t / tau_0) + b_1 * exp(-t / tau_1)

    where ``tau_0 > tau_1``.

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
    e = expfit.MultiExponentialError(t, v, 1, 1, p0[1] > 0)
    q0 = e.transform(p0)
    with np.errstate(all='ignore'):
        r = expfit.lm(e, q0, plot=opt_plot)
        if plot is not False:  # pragma: no cover
            print(r)

    # Detransform parameters
    et = expfit.TauFormError(t, v)
    p = expfit.ExponentialFit(t, v, e.detransform(r.x, True), et)

    if plot is not False:  # pragma: no cover
        from ._plot import tau_plot
        p0 = expfit.ExponentialFit(t, v, e.detransform(q0, True), et)
        pt = None
        try:
            assert len(plot) == 5
            pt = plot
        except (TypeError, AssertionError):
            pass
        tau_plot(t, v, r, p, p0, pt=pt)

    return p

