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
        Optional parameter to create a plot of the method's workings.

    Returns an :class:`ExponentialFit`.
    """
    t, v = expfit.vet_series(t, v)

    # Transform to unit square, to avoid overflows etc
    tr = expfit.UnitSquareTransform(t, v)

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
        fig = plt.figure(figsize=(9, 7.5))
        ax0 = fig.add_subplot(2, 1, 1)
        fig.subplots_adjust(0.11, 0.06, 0.995, 0.995, wspace=0.3, hspace=0.44)
        ax0.set_xlabel('x')
        ax0.set_ylabel('y')
        ls, color = ('-', '#92cc92') if len(tr.x) > 10 else ('x-', 'tab:green')
        ax0.plot(tr.x, tr.y, ls, color=color, label='Transformed data')
    else:
        ax0 = None

    # Get an initial estimate (in transformed space)
    #q0 = expfit.estimate_initial_single(tr.x, tr.y, axes=ax0, vet=False)
    #TODO
    #TODO
    q0 = expfit.estimate_initial_single(tr.x, tr.y, vet=False)
    #TODO
    #TODO

    # Stop if the signal is not exponential
    if q0[1] == 0 or q0[2] == 0:
        raise expfit.NotExponentialError()

    # Fit (in transformed space)
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
        # Detransform and rewrite in tau form
        p0 = tr.detransform(q0)
        p0[2] = -1 / p0[2]
        p0 = expfit.ExponentialFit(t, v, p0)
        q0 = list(q0)
        q0[2] = -1 / q0[2]
        q0 = expfit.ExponentialFit(tr.x, tr.y, q0)
        q = r.x
        q[2] = -1 / q[2]
        q = expfit.ExponentialFit(tr.x, tr.y, r.x)

        # Create strings for plot labels
        strest = ', '.join(f'{i:.3}' for i in q0)
        strq = ', '.join(f'{i:.3}' for i in q)
        stre = f'rmse {np.sqrt(r.error):.4}'
        if r.success:
            strfit = f'{r.iterations} iter, {stre}'
        else:
            strfit = f'{r.message}, {stre}'

        # Plot initial estimate and fit
        e = expfit.exp
        ax0.plot(tr.x, e(tr.x, q0), '-', label=f'Initial ({strest})')
        ax0.plot(tr.x, e(tr.x, q), '--', label=f'Fit ({strq}), {strfit}')
        ax0.legend()

        # Plot numerical results
        lines = [f'Transformed Init: {q0}', f'             Fit:  {q}',
                 f'Real-world  Init: {p0}', f'             Fit:  {p}']
        ax0.text(0.75, -0.38, '\n'.join(lines), transform=ax0.transAxes,
                 ha='right', font='monospace')

        # Show the residuals for initial estimate and fit
        ax1 = fig.add_subplot(2, 2, 3)
        ax1.set_xlabel('x')
        ax1.set_ylabel('Residuals (transformed)')
        ax1.plot(tr.x, tr.y - e(tr.x, q0), label='Initial')
        ax1.plot(tr.x, tr.y - e(tr.x, q), label='Fit')
        ax1.legend()

        # Show untransformed, including the original data
        ax2 = fig.add_subplot(2, 2, 4)
        ax2.set_xlabel('t')
        ax2.set_ylabel('v')
        label = 'Original data'
        with np.errstate(divide='ignore'):
            if known:
                label = f'{label} (tau={known[2]:+.3f})'
            ax2.plot(t, v, ls, color=color, label=label)
            ax2.plot(t, e(t, p0), '-', label=f'Initial (tau={p0[2]:+.3f})')
            ax2.plot(t, e(t, p), '--', label=f'fFit (tau={p[2]:+.3f})')
        ax2.legend()

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
            if plot is not False:  # pragma: no cover
                print(r)
            opt_fig = r.plot

        #print(r.x)
        if np.exp(r.x[4] - r.x[2]) > 1.1 and r.success:
            break
        elif i + 1 == max_iter:  # pragma: no cover
            raise RuntimeError(
                f'Unable to find good fit after {max_iter} attempts.')

    # Detransform parameters
    et = expfit.TauFormError(t, v)
    p = expfit.ExponentialFit(t, v, e.detransform(r.x, True), et)

    #print(f'Done in {1 + i} repeats. Last opt had {r.iterations} iter.')
    if plot is not False:  # pragma: no cover
        from ._plot import tau_plot

        p0 = expfit.ExponentialFit(t, v, e.detransform(q0, True), et)
        pk = None
        try:
            assert len(plot) == 5
            pk = plot
        except (TypeError, AssertionError):
            pass

        fig, (ax, iax, tax) = tau_plot(t, v, r, p, p0, pk)
        iax[0].plot(t, expfit.expc(t, (a0, b0, c0)), 'k--', lw=1.5,
                    label=f'Single ($\\tau$={-1 / c0:.3g})')
        iax[0].legend(frameon=False)

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
        tau_plot(t, v, r, p, p0, pt)

    return p

