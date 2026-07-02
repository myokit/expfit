#
# Single and multi-expontial fits
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import numpy as np

import expfit


def fit1(x, y=None, plot=False, opt_plot=False):
    """
    Fits an exponential ``a + b * exp(c * x)`` to the time series ``(x, y)``,
    returning ``(a, b, c)``

    Example::

        x = np.linspace(0, 1, 100)
        y = 3 + 2 * np.exp(-5 * x) + np.random.normal(0, 1, size=len(t))
        a, b, c = expfit.fit_single(x, y)
        print(a, b, c)

    Arguments:

    ``x``, ``y``
        The time series as two one-dimensional arrays of equal size.
        Alternatively, ``x, y`` can be a :class:`TimeSeries` and ``None``.
    ``plot``
        Optional parameter to create a plot of the method's workings. Can be
        set to ``True`` or to an array with the true ``(a, b, c)``.
    ``opt_plot``
        Optional parameter to create a plot of the optimisation routine.

    Returns a tuple ``(a, b, c)``.
    """
    xy = xy_org = expfit.TimeSeries._from_xy(x, y)
    if not isinstance(xy_org, expfit.UnitSquaredSeries):
        xy = expfit.UnitSquaredSeries(*xy)
    del x, y

    # Convert `plot` to boolean
    pt = plot
    plot = plot is not False

    # Get an initial estimate in transformed space
    # May raise a NotExponentialError
    q0 = expfit.estimate_initial_single(xy, full=plot)

    # Fit
    e = expfit.SingleExponentialError(xy)
    with np.errstate(all='ignore'):
        r = expfit.lm(e, q0, plot=opt_plot)
        if plot:  # pragma: no cover
            print(r)

    # Create result object with CI capabilities, on original data
    p = expfit.ExponentialFit(
        xy.detransform(r.x), expfit.SingleExponentialError(xy_org))

    # Plot, especially when not successful
    if plot:  # pragma: no cover
        from ._plot import fit1_plot
        try:
            assert len(pt) == 3
        except (TypeError, AssertionError):
            pt = None
        fit1_plot(xy, q0, r, xy_org, p, pt)

    # Fail if optimisation failed, but still provide parameters
    if not r.success:
        raise expfit.FitFailedError(
            f'Fit failed with optimiser message: {r.message}', r, p)

    # Create CI-enabled parameter set and return
    return p


def fitd2(x, y, plot=False, opt_plot=False):
    """
    Fits a decaying double-exponential to a time series, with equal signed
    multipliers for both components.

    Returns parameters for::

        y = a + b_0 * exp(-t / tau_0) + b_1 * exp(-t / tau_1)

    where ``tau_0 > tau_1``.

    Arguments:

    ``x``, ``y``
        The time series as two one-dimensional arrays of equal size.
        Alternatively, ``x, y`` can be a :class:`TimeSeries` and ``None``.
    ``plot``
        Optional parameter to create a plot showing the final results,
        including confidence intervals on the time constants.
    ``opt_plot``
        Optional parameter to create a plot of the optimisation routine.

    Returns an :class:`ExponentialFit`.
    """
    x, y = xy = expfit.TimeSeries._from_xy(x, y)

    # Convert `plot` to boolean
    pt = plot
    plot = plot is not False

    # Estimate the dominant rate on unit-transformed data
    tr = xy
    if not isinstance(tr, expfit.UnitSquaredSeries):
        tr = expfit.UnitSquaredSeries(*xy)
    q0 = expfit.estimate_initial_single(tr)
    a0, b0, c0 = tr.detransform(q0)
    del tr, q0

    # Stop if not decaying
    if c0 > 0:
        raise expfit.NotDecayingError()

    # Fit double (in untransformed space)

    # Calculate area (ignoring a), to determine new b constants
    A0 = (b0 / c0) * (np.exp(c0 * x[-1]) - np.exp(c0 * x[0]))

    # Assume dominant (slowest) rate found, next will be faster
    p0 = np.array((a0, b0, c0, b0, c0), dtype=float)
    p0[1] *= 0.7    # The second exponential will contribute
    p0[2] *= 0.5    # The first c will be overestimated

    # Set up error
    e = expfit.MultiExponentialError(xy, 2, 0, b0 > 0)

    max_iter = 10
    opt_fig = opt_plot
    for i in range(max_iter):
        # Speed up the second exponential
        p0[4] *= 1.4

        # Set b constants to get same area under the curve as original estimate
        A1 = p0[1] / p0[2] * (np.exp(p0[2] * x[-1]) - np.exp(p0[2] * x[0]))
        A2 = p0[3] / p0[4] * (np.exp(p0[4] * x[-1]) - np.exp(p0[4] * x[0]))
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
    p_tau = e.detransform(r.x, tau=True)
    e_tau = expfit.TauFormError(xy)
    p = expfit.ExponentialFit(p_tau, e_tau)

    if plot:  # pragma: no cover
        from ._plot import tau_plot
        p0[2::2] = -1 / p0[2::2]
        try:
            assert len(pt) == 5
        except (TypeError, AssertionError):
            pt = None
        tau_plot(xy, p0, r, p, pt)

    return p


def fitd11(x, y=None, plot=False, opt_plot=False):
    """
    Fits a decaying double-exponential to a time series, with opposite signed
    multipliers for both components.

    Returns parameters for::

        v = a + b_0 * exp(-t / tau_0) + b_1 * exp(-t / tau_1)

    where ``tau_0 > tau_1``.

    Arguments:

    ``x``, ``y``
        The time series as two one-dimensional arrays of equal size.
        Alternatively, ``x, y`` can be a :class:`TimeSeries` and ``None``.
    ``plot``
        Optional parameter to create a plot showing the final results,
        including confidence intervals on the time constants.
    ``opt_plot``
        Optional parameter to create a plot of the optimisation routine.

    Returns an :class:`ExponentialFit`.
    """
    x, y = xy = expfit.TimeSeries._from_xy(x, y)

    # Convert `plot` to boolean
    pt = plot
    plot = plot is not False

    # Perform initial estimates on unit-transformed data
    tr = xy
    if not isinstance(tr, expfit.UnitSquaredSeries):
        tr = expfit.UnitSquaredSeries(*xy)
    q0 = expfit.estimate_initial_opposing(tr)
    p0 = tr.detransform(q0)
    del tr, q0

    # Fit double
    e = expfit.MultiExponentialError(xy, 1, 1, p0[1] > 0)
    q0 = e.transform(p0)
    with np.errstate(all='ignore'):
        r = expfit.lm(e, q0, plot=opt_plot)
        if plot:  # pragma: no cover
            print(r)

    # Detransform parameters
    p_tau = e.detransform(r.x, tau=True)
    e_tau = expfit.TauFormError(xy)
    p = expfit.ExponentialFit(p_tau, e_tau)

    if plot:  # pragma: no cover
        from ._plot import tau_plot
        p0[2::2] = -1 / p0[2::2]
        try:
            assert len(pt) == 5
        except (TypeError, AssertionError):
            pt = None
        tau_plot(xy, p0, r, p, pt=pt)

    return p


def auto(t, v, plot=False, opt_plot=False):
    """
    """
    t, v = expfit.vet_series(t, v)

    # Convert `plot` to boolean
    pt = plot
    plot = plot is not False

    # Transform to unit square
    tr = expfit.UnitSquareTransform(t, v)

    # TODO: Do everything in transformed space?
    q0 = expfit.estimate_initial_single(tr.x, tr.y, vet=False, plot=True)
    p0 = tr.detransform(q0)
    del tr, q0

    # Catch edge cases
    #if p0[1] * p0[3] >= 0:
    #    raise expfit.NotOpposingError()
    #if p0[2] > 0 or p0[4] > 0:
    #    raise expfit.NotDecayingError()

    nd = 1
    no = 0
    dom_pos = p0[1] > 0

    # Fit single
    e = expfit.MultiExponentialError(t, v, nd, no, dom_pos)
    q0 = e.transform(p0)
    print('Start', p0)
    print('Start', q0)
    with np.errstate(all='ignore'):
        r = expfit.lm(e, q0)
        if plot:  # pragma: no cover
            print(r)
        #if not r.success:
        #    p0[2] *= 0.1
        #    q0 = e.transform(p0)
        #    r = expfit.lm(e, q0)
        #    if plot:  # pragma: no cover
        #        print(r)

    a0, b0, c0 = e.detransform(r.x, tau=False)

    # Store p_best in tau form
    p_best = (a0, b0, -1 / c0)
    E_best = r.error
    p0_best = (p0[0], p0[1], -1 / p0[2])
    p0_next = p0

    # Estimate sigma
    s = expfit.estimate_noise_level(t, v, vet=False)

    # Calculate required improvement to accept
    w = s**2 / len(t) * expfit.CLevel(90).chi2()
    print('Required improvement: ', w)

    # Calculate area, to determine new b constants
    #A0 = (b0 / c0) * (np.exp(c0 * t[-1]) - np.exp(c0 * t[0]))
    for i in range(4):
        nd += 1
        e = expfit.MultiExponentialError(t, v, nd, no, dom_pos)
        c = expfit.MultiExponentialConstraint()

        p0 = np.zeros(1 + 2 * nd)
        p0[0] = p0_next[0]
        p0[1:-2:2] = p0_next[1::2]
        p0[2:-2:2] = p0_next[2::2]
        p0[-2:] = p0_next[-2:]

        print('-' * 70)
        print(f'Trying with {nd} terms')
        print('-' * 70)

        max_iter = 10
        opt_fig = opt_plot
        for j in range(max_iter):
            print(p0)

            p0[2] *= 0.7
            for k in range(2, nd + 1):
                p0[2 * k] *= 1.4**(k / 2)
            print(p0)
            print()

            # Set initial b to get same area under the curve as single exp fit
            #A = 0
            #for k in range(nd):
            #    b, c = p0[1 + 2 * k], p0[2 + 2 * k]
            #    A += b / c * (np.exp(c * t[-1]) - np.exp(c * t[0]))
            #p0[1::2] = b0 * (A0 / A)

            # Fit with transformed parameters
            q0 = e.transform(p0)
            with np.errstate(all='ignore'):
                r = expfit.lm(e, q0, constraint=c, plot=opt_fig)
                if plot:  # pragma: no cover
                    print(r)
                opt_fig = r.plot
            ok = r.success
            if ok:
                for k in range(nd - 1):
                    if np.exp(r.x[4 + 2 * k] - r.x[2 + 2 * k]) <= 1.1:
                        print('TOO CLOSE', r.x[4 + 2 * k], r.x[2 + 2 * k])
                        ok = False
                        break
            if ok:
                break
        if j + 1 == max_iter:  # pragma: no cover
            print(f'Unable to find good fit after {max_iter} attempts.')

        print()
        print('Error', r.error)
        print('Ebest', E_best)
        print('Improvement', E_best - r.error)
        print('Required   ', w)

        if E_best - r.error > w:
            p_best = e.detransform(r.x, tau=True)
            E_best = r.error
            p0_best = e.detransform(q0, tau=True)
            p0_next = e.detransform(q0, tau=False)
        else:
            break

    # Create CI object
    et = expfit.TauFormError(t, v)
    p = expfit.ExponentialFit(t, v, p_best, et)

    if plot:  # pragma: no cover
        from ._plot import tau_plot
        try:
            assert len(pt) % 2 == 1
        except (TypeError, AssertionError):
            pt = None
        tau_plot(t, v, r, p, p0_best, pt=pt)

