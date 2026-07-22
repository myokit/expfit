#
# Single and multi-expontial fits
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import numpy as np

import expfit


def fit1(x, y=None, plot=False, opt_plot=False, full=False):
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
    ``full``
        Set to ``True`` to return a list containing the initial estimate at
        index 0, followed by the obtained fit.

    Returns an :class:`expfit.ExponentialFit` (or a list if ``full=True``).
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
    q0 = expfit.est1(xy, full=plot)

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

    # Return
    return [xy.detransform(q0), p] if full else p


# TODO remove TODO remove TODO remove TODO remove TODO remove TODO remove TODO
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
    q0 = expfit.est1(tr)
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
# TODO remove TODO remove TODO remove TODO remove TODO remove TODO remove TODO


def fitd11(x, y=None, sigma=None, plot=False, opt_plot=False, full=False):
    """
    Fits a decaying double-exponential to a time series, with opposite signed
    multipliers for both components.

    Returns parameters for::

        v = a + b_0 * exp(-t / tau_0) + b_1 * exp(-t / tau_1)

    where ``tau_0 > tau_1``.

    The method requires an estimate of the standard deviation of the noise
    (under the assumption of Gaussian noise). This can often be obtained from
    a flat bit of signal. If not given, it will be determined from the last
    10% of the signal using :meth:`expfit.estimate_noise_level()`.

    Arguments:

    ``x``, ``y``
        The time series as two one-dimensional arrays of equal size.
        Alternatively, ``x, y`` can be a :class:`TimeSeries` and ``None``.
    ``sigma``
        An estimate of the noise level in the data, given as a standard
        deviation.
    ``plot``
        Optional parameter to create a plot showing the final results,
        including confidence intervals on the time constants.
    ``opt_plot``
        Optional parameter to create a plot of the optimisation routine.
    ``full``
        Set to ``True`` to return a list containing the initial estimate at
        index 0, followed by the obtained fit.

    Returns an :class:`ExponentialFit` (or a list if ``full=True``).
    """
    x, y = xy = expfit.TimeSeries._from_xy(x, y)

    # Convert `plot` to boolean
    pt = plot
    plot = plot is not False

    # Perform initial estimates on unit-transformed data
    tr = xy
    if not isinstance(tr, expfit.UnitSquaredSeries):
        tr = expfit.UnitSquaredSeries(*xy)

    # TODO, unit transform sigma and pass in

    q0 = expfit.estd11(tr)
    p0 = tr.detransform(q0)
    del tr, q0

    # Fit double
    e = expfit.MultiExponentialError(xy, 1, 1, p0[1] > 0)
    q0 = e.transform(p0, tau=False)
    with np.errstate(all='ignore'):
        r = expfit.lm(e, q0, plot=opt_plot)
        if plot:  # pragma: no cover
            print(r)

    # Detransform parameters
    p_tau = e.detransform(r.x, tau=True)
    e_tau = expfit.TauFormError(xy)
    p = expfit.ExponentialFit(p_tau, e_tau)

    # TODO
    # Check if a single exponential fits better.
    # Check if a straight line fits better.
    # No longer done as part of estd11

    if plot:  # pragma: no cover
        from ._plot import tau_plot
        p0[2::2] = -1 / p0[2::2]
        try:
            assert len(pt) == 5
        except (TypeError, AssertionError):
            pt = None
        tau_plot(xy, p0, r, p, pt=pt)

    return [p, p0] if full else p


def auto(x, y, plot=False, opt_plot=False):
    """
    """
    x, y = xy = expfit.TimeSeries._from_xy(x, y)

    # Convert `plot` to boolean, set pt to None if wrong size
    pt = plot
    plot = plot is not False
    if plot:
        try:
            assert len(pt) % 2 == 1
        except (TypeError, AssertionError):
            pt = None

    # Create error in tau form, for use in returned CI objects
    etau = expfit.TauFormError(xy)

    # Test if oppsing signs are present, and get initial estimates if so
    opposing = False
    try:
        p, p0 = expfit.fitd11(xy, plot=False, full=True)
    except expfit.NotOpposingError:
        pass
    else:
        opposing = True

    # If opposing: start from a double exponential fit
    if opposing:
        # Already in tau form

        # Set up first problem
        nd = no = 1
        dom_pos = p0[3] > 0

        # Store solution
        solutions = [expfit.ExponentialFit(p, etau)]
        if plot:
            # Store initial guess and optimiser results, if available
            tau_plot_info = [(p0, None)]

        #raise NotImplementedError


    # Not opposing: start from a single exponential fit
    else:
        # Fit single in c form and convert to tau form
        p, p0 = expfit.fit1(xy, full=True)
        p = p[0], p[1], -1 / p[2]
        p0 = p0[0], p0[1], -1 / p0[2]

        # Stop if not decaying
        if p[2] < 0:
            raise expfit.NotDecayingError()

        # Set up first problem
        nd = 1
        no = 0
        dom_pos = p0[1] > 0

        # Store solution
        solutions = [expfit.ExponentialFit(p0, etau)]
        if plot:
            # Store initial guess and optimiser results, if available
            tau_plot_info = [(p0, None)]

    # Best solution and error
    pbest = solutions[-1]
    ebest = pbest.mse()

    # TODO IMPROVE THIS
    # Estimate sigma
    s = expfit.estimate_noise_level(x, y)

    # Calculate required improvement to accept
    w = s**2 / len(x) * expfit.CLevel(90).chi2()
    print('Required improvement: ', w)

    # Calculate area, to determine new b constants
    #A0 = (b0 / c0) * (np.exp(c0 * t[-1]) - np.exp(c0 * t[0]))

    '''

    # Try up to 4 exponentials
    for i in range(10):  # TODO
        nd += 1  # TODO
        e = expfit.MultiExponentialError(xy, nd, no, dom_pos)
        c = expfit.MultiExponentialConstraint()



        p0 = np.zeros(1 + 2 * nd)
        p0[:-2] = pbest
        p0[-2:] = pbest[-2:]

        print('-' * 70)
        print(f'Trying with {nd} terms')
        print('-' * 70)

        max_iter = 10
        opt_fig = opt_plot
        for j in range(max_iter):
            print('Iteration', j)
            print(p0)

            p0[2] *= 1.4
            for k in range(2, nd + 1):
                p0[2 * k] *= 0.7**(k / 2)
            print(p0)
            print()

            # Set initial b to get same area under the curve as single exp fit
            #A = 0
            #for k in range(nd):
            #    b, c = p0[1 + 2 * k], p0[2 + 2 * k]
            #    A += b / c * (np.exp(c * t[-1]) - np.exp(c * t[0]))
            #p0[1::2] = b0 * (A0 / A)

            # Fit with transformed parameters
            q0 = e.transform(p0, tau=True)
            with np.errstate(all='ignore'):
                r = expfit.lm(e, q0, constraint=c, plot=opt_fig)
                if plot:  # pragma: no cover
                    print(r)
                opt_fig = r.plot
            ok = r.success
            if ok:
                for k in range(nd - 1):
                    # TODO UPDATE THIS TODO TODO TODO TODO TODO TODO TODO
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
        print('Ebest', ebest)
        print('Improvement', ebest - r.error)
        print('Required   ', w)

        if ebest - r.error > w:
            pbest = e.detransform(r.x, tau=True)
            ebest = r.error
            solutions.append(expfit.ExponentialFit(pbest, etau))
            if plot:
                tau_plot_info.append((p0, r))
        else:
            break

    '''

    # Create plots
    if plot:  # pragma: no cover
        # FIM plot
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot()
        for p in solutions:
            c = None
            d = (len(p) - 1) // 2
            q = 0.5 / np.diag(np.linalg.inv(p.hes()))
            label = f'FIM, {d} exp'
            for i in range(d):
                j = 2 + 2 * i
                lo, hi = p.ci_fisher(j)
                x = np.linspace(lo, hi, 100)
                c = ax.plot(x, p.mse() + q[j] * (x - p[j])**2,
                            label=label, color=c)[0].get_color()
                label = None
        ax.legend()

        # Tau plots
        from ._plot import tau_plot
        for p, (p0, r) in zip(solutions, tau_plot_info):
            tau_plot(xy, p0, r, p, pt)



    # Return best fit

