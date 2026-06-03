#
# Single and multi-expontial fits
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import numpy as np

import expfit


C1 = 'tab:red'
D1 = '#961b1c'
C2 = 'tab:purple'
D2 = '#5b3383'
# '#1f701f'


def fit1(t, v, plot=False):
    """
    Fits an exponential ``a + b * exp(c * t)`` to the time series ``(t, v)``,
    returning ``(a, b, c)``

    Example::

        t = np.linspace(0, 1, 100)
        v = 3 - 2 * np.exp(4 * t) + np.random.normal(0, 1, size=len(t))
        a, b, c = expfit.fit_single(t, v)
        print(a, b, c)

    Arguments:

    ``t``, ``v``
        The time series
    ``plot``
        Optional parameter to create a plot of the method's workings. Can be a
        boolean or the string "simple" for a reduced plot.

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
        if plot == 'simple':
            fig = plt.figure(figsize=(8, 4))
            ax0 = fig.add_subplot()
        else:
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
    at0, bt0, ct0 = expfit.estimate_initial_single(
        tr.x, tr.y, axes=ax0, vet=False)

    # Fit (in transformed space)
    e = expfit.SingleExponentialError(tr.x, tr.y)
    with np.errstate(all='ignore'):
        r = expfit.fmin(e, (at0, bt0, ct0))
        if plot:  # pragma: no cover
            print(r)
    at, bt, ct = r.x

    # Detransform obtained parameters, create result
    p = ExponentialFit(t, v, tr.detransform(at, bt, ct))

    if plot:  # pragma: no cover
        p0 = ExponentialFit(t, v, tr.detransform(at0, bt0, ct0))
        q0 = ExponentialFit(tr.x, tr.y, (at0, bt0, ct0))
        q = ExponentialFit(tr.x, tr.y, (at, bt, ct))

        strest = ', '.join(f'{i:.3}' for i in q0)
        strq = ', '.join(f'{i:.3}' for i in q)
        stre = f'rmse {np.sqrt(r.error):.4}'
        if r.success:
            strfit = f'{strq}, {r.iterations} iter, {stre}'
        else:
            strfit = f'{strq}, {r.message}, {stre}'

        e = expfit.exp
        ax0.plot(tr.x, e(tr.x, q0), '-', label=f'Initial ({strest})')
        ax0.plot(tr.x, e(tr.x, q), '--', label=f'Fit ({strfit})')
        ax0.legend()

        if plot != 'simple':
            lines = [f'Transformed Init: {q0}', f'             Fit:  {q}',
                     f'Real-world  Init: {p0}', f'             Fit:  {p}']
            ax0.text(0.75, -0.38, '\n'.join(lines), transform=ax0.transAxes,
                     ha='right', font='monospace')

            ax1 = fig.add_subplot(2, 2, 3)
            ax1.set_xlabel('x')
            ax1.set_ylabel('Residuals (transformed)')
            ax1.plot(tr.x, tr.y - e(tr.x, q0), label='Initial')
            ax1.plot(tr.x, tr.y - e(tr.x, q), label='Fit')
            ax1.legend()

            ax2 = fig.add_subplot(2, 2, 4)
            ax2.set_xlabel('t')
            ax2.set_ylabel('v')
            label = 'Untransformed data'
            with np.errstate(divide='ignore'):
                if known:
                    label = f'{label} (tau={-1 / known[2]:+.3f})'
                ax2.plot(t, v, ls, color=color, label=label)
                strc0 = f'c={p0[2]:+.3f}, tau={-1 / p0[2]:+.3f}'
                strc = f'c={p[2]:+.3f}, tau={-1 / p[2]:+.3f}'
                ax2.plot(t, e(t, p0), '-', label=f'Initial ({strc0})')
                ax2.plot(t, e(t, p), '--', label=f'fFit ({strc})')
            ax2.legend()

    return p


def fitd2(t, v, plot=False):
    """
    Fits a double-exponential ``y = a + b0 * exp(c0 * x) + b1 * exp(c1 * x)``,
    where ``b0`` and ``b1`` have the same sign, ``c0`` and ``c1`` are both
    negative, and ``c1 > c0``.

    Arguments:

    ``t``, ``v``
        The time series
    ``plot``
        Optional parameter to create a plot of the method's workings. Can be a
        boolean or an array of known (true) parameters.

    Returns an :class:`ExponentialFit`.
    """
    t, v = expfit.vet_series(t, v)

    # Estimate the dominant rate (in transformed space)
    tr = expfit.UnitSquareTransform(t, v)
    q0 = expfit.estimate_initial_single(tr.x, tr.y, vet=False)
    a0, b0, c0 = tr.detransform(q0)
    del tr, q0

    # Avoid nans etc.
    if c0 == 0:
        return ExponentialFit(t, v, (a0, b0, 0, 0, 0))

    # Catch non-decaying
    if c0 > 0:
        raise RuntimeError(
            'Initial estimate for c > 0, exponential not decaying')

    # Fit double (in untransformed space)
    # Assume dominant (slowest) rate found, next will be faster
    p0 = np.array((a0, b0, c0, b0, c0), dtype=float)
    ct = expfit.DecayingConstraint()
    for i in range(1, 10):
        p0[2] *= 0.707106781
        p0[4] *= 1.414213562
        e = expfit.MultiExponentialError(t, v)
        with np.errstate(all='ignore'):
            r = expfit.fmin(e, p0, constraint=ct)
            if plot:  # pragma: no cover
                print(r)
        if r.x[4] / r.x[2] > 1.1 and r.success:
            break

    p = ExponentialFit(t, v, r.x, constraint=ct)

    if plot:  # pragma: no cover
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(11, 7.5))
        fig.subplots_adjust(0.075, 0.06, 0.99, 0.95, wspace=0.37, hspace=0.3)
        grd = fig.add_gridspec(3, 3, width_ratios=(2, 2, 1))

        # Show data
        code = '-' if len(t) > 10 else 'x-'
        ax0 = fig.add_subplot(grd[:2, :2])
        ax0.set_xlabel('t')
        ax0.set_ylabel('v')
        ax0.plot(t, v, code, color='tab:blue', label=f'Data (n={len(t)})')

        # Show parameters
        p0 = ExponentialFit(t, v, p0)
        ax0.text(0.5, 1.015, f'Init: {p0}\n Fit: {p}',
                 transform=ax0.transAxes, ha='center', font='monospace')

        # Try showing known solution
        e = expfit.exp
        known = False
        try:
            assert len(plot) == 5
            known = True
        except (TypeError, AssertionError):
            pass
        if known:
            ax0.plot(t, e(t, (plot[0], plot[1], plot[2])), color=C1,
                     label=f'Known 1st (tau={-1 / plot[2]:.2g})',)
            ax0.plot(t, e(t, (plot[0], plot[3], plot[4])), color=C2,
                     label=f'Known 2nd (tau={-1 / plot[4]:.2g})')

        # Show fit
        label = f'rmse {np.sqrt(r.error):.4}'
        if r.success:
            label = f'Fit ({r.iterations} iter, {label})'
        else:
            label = f'Fit ({r.message}, {label})'
        ax0.plot(t, e(t, p), lw=1, color='k', label=label)

        # First exponential
        lo1, hi1 = p.ci(2)
        cif1 = p.ci_fisher(2)
        tau1, tlo1, thi1 = -1 / p[2], -1 / hi1[2], -1 / lo1[2]
        tclo1, tchi1 = -1 / (p[2] - cif1), -1 / (p[2] + cif1)
        b = (f'Fit 1st (tau={tau1:.2g}, PL[{tlo1:.2g}, {thi1:.2g}],'
             f' FI[{tclo1:.2g}, {tchi1:.2g}])')
        ax0.plot(t, e(t, (p[0], p[1], p[2])), lw=1, ls='--', color=D1, label=b)
        ax0.fill_between(t, e(t, (lo1[0], lo1[1], lo1[2])),
                         e(t, (hi1[0], hi1[1], hi1[2])), color=D1, alpha=0.1)
        ax0.plot(t, e(t, (lo1[0], lo1[1], lo1[2])), lw=0.4, color=D1)
        ax0.plot(t, e(t, (hi1[0], hi1[1], hi1[2])), lw=0.4, color=D1)
        ax0.plot(t, e(t, lo1), 'tab:green', ls='--', lw=0.4)
        ax0.plot(t, e(t, hi1), 'tab:green', ls='--', lw=0.4)

        # Second exponential
        lo2, hi2 = p.ci(4)
        cif2 = p.ci_fisher(4)
        tau2, tlo2, thi2 = -1 / p[4], -1 / hi2[4], -1 / lo2[4]
        tclo2, tchi2 = -1 / (p[4] - cif2), -1 / (p[4] + cif2)
        b = (f'Fit 1st (tau={tau1:.2g}, PL[{tlo1:.2g}, {thi1:.2g}],'
             f' FI[{tclo1:.2g}, {tchi1:.2g}])')
        b = (f'Fit 2nd (tau={tau2:.2g} PL[{tlo2:.2g}, {thi2:.2g}],'
             f' FI[{tclo2:.2g}, {tchi2:.2g}])')
        ax0.plot(t, e(t, (p[0], p[3], p[4])), lw=1, ls='--', color=D2, label=b)
        ax0.fill_between(t, e(t, (lo2[0], lo2[3], lo2[4])),
                         e(t, (hi2[0], hi2[3], hi2[4])), color=D2, alpha=0.1)
        ax0.plot(t, e(t, (lo2[0], lo2[3], lo2[4])), lw=0.4, color=D2)
        ax0.plot(t, e(t, (hi2[0], hi2[3], hi2[4])), lw=0.4, color=D2)
        ax0.plot(t, e(t, lo2), 'tab:green', ls='--', lw=0.4)
        ax0.plot(t, e(t, hi2), 'tab:green', ls='--', lw=0.4)

        # Finalise main panel
        ax0.legend(framealpha=1, ncol=2)

        # Show single exponential estimate
        ax1 = fig.add_subplot(grd[2, 0])
        ax1.set_xlabel('t')
        ax1.set_ylabel('v')
        ax1.plot(t, v, code, color='tab:blue', label='Data')
        ax1.plot(t, e(t, (a0, b0, c0)), 'k--', lw=1.5,
                 label=f'Init. single ($\\tau$={-1 / c0:.2g})')
        ax1.plot(t, e(t, p0), 'k:', lw=1.5,
                 label='Init. double')
        ax1.legend(frameon=False)

        # Show final fit residuals
        ax2 = fig.add_subplot(grd[2, 1])
        ax2.set_xlabel('t')
        ax2.set_ylabel('Residuals')
        ax2.plot(t, v - e(t, p))

        # Show MSE profile for tau 1
        ax4 = fig.add_subplot(grd[0, 2])
        ax4.set_xlabel('tau 1')
        ax4.set_ylabel('MSE')
        values, errors = p.pl(2, lo1[2], hi1[2])
        ax4.plot(-1 / values, errors)
        ax4.axvline(tau1, color='gray', zorder=1)
        ax4.axvline(tlo1, color='k', lw=1)
        ax4.axvline(thi1, color='k', lw=1)
        ax4.axvline(tclo1, color='k', lw=1, ls='--')
        ax4.axvline(tchi1, color='k', lw=1, ls='--')

        # Add parabola based on Hessian
        x = np.linspace(values[0], values[-1], 100)
        a = r.hes[2, 2] * 0.5
        b = r.jac[2] - 2 * a * p[2]
        c = r.error - p[2] * (b + a * p[2])
        ylim = ax4.get_ylim()
        ax4.plot(-1 / x, a * x**2 + b * x + c, '-')
        ax4.set_ylim(ylim)
        xticks = np.array([values[0], values[len(values) // 2], values[-1]])
        ax4.set_xticks(-1 / xticks)

        # Show MSE profile for tau 2
        ax5 = fig.add_subplot(grd[1, 2])
        ax5.set_xlabel('tau 2')
        ax5.set_ylabel('MSE')
        values, errors = p.pl(4, lo2[4], hi2[4])
        ax5.plot(-1 / values, errors)
        ax5.axvline(-1 / p[4], color='gray', zorder=1)
        ax5.axvline(tlo2, color='k', lw=1)
        ax5.axvline(thi2, color='k', lw=1)
        ax5.axvline(tclo2, color='k', lw=1, ls='--')
        ax5.axvline(tchi2, color='k', lw=1, ls='--')

        # Add parabola based on Hessian
        x = np.linspace(values[0], values[-1], 100)
        a = r.hes[4, 4] * 0.5
        b = r.jac[4] - 2 * a * p[4]
        c = r.error - p[4] * (b + a * p[4])
        ylim = ax5.get_ylim()
        ax5.plot(-1 / x, a * x**2 + b * x + c, '-')
        ax5.set_ylim(ylim)
        ax4.set_xticks(-1 / xticks)

        # Show error comparison with known
        if known:
            ax3 = fig.add_subplot(grd[2, 2])
            found_vs_true(ax3, expfit.MultiExponentialError(t, v), p, plot)
            fig.align_ylabels((ax3, ax4, ax5))
        else:
            fig.align_ylabels((ax3, ax4))

        '''
        # Show covariance matrices
        ax3 = fig.add_subplot(grd[1, 2])

        cov = p.cov()
        cv = cov[2::2, 2::2]
        cov_ellipse(ax3, p[2::2], cv)
        xlim, ylim = ax3.get_xlim(), ax3.get_ylim()
        rx, ry = xlim[1] - xlim[0], ylim[1] - ylim[0]
        if rx > ry:
            m = ylim[1] + ylim[0]
            ax3.set_ylim(0.5 * (m - rx), 0.5 * (m + rx))
        else:
            m = xlim[1] + xlim[0]
            print(0.5 * m)
            ax3.set_xlim(0.5 * (m - ry), 0.5 * (m + ry))
        ax3.set_xlabel('c1')
        ax3.set_ylabel('c2')

        err = np.diag(cov)
        t1 = 1.96 * np.sqrt(err[2])
        t2 = 1.96 * np.sqrt(err[4])
        print()
        print(t1)
        print(t2)
        print()
        print(1 / t1)
        print(1 / t2)
        print()
        print(-1 / (p[2] - t1), -1 / (p[2] + t1))
        print(-1 / (p[4] - t2), -1 / (p[4] + t2))
        print()
        #'''

    return p


def found_vs_true(ax, error, found, known, padding=0.25, evaluations=200):
    """
    Plots the RMSE between a ``found`` and ``known`` (true) value.
    """
    found, known = np.array(found), np.array(known)
    f = lambda p: np.sqrt(error(p)[0])
    s = np.linspace(-padding, 1 + padding, evaluations)
    r = known - found
    x = [found + sj * r for sj in s]
    y = [np.sqrt(f(i)) for i in x]
    ax.plot(s, y, color='green')
    ax.axvline(0, color='#1f77b4', label='Found')
    ax.axvline(1, color='#7f7f7f', label='True')
    ax.set_ylabel('RMSE')
    ax.legend()


def cov_ellipse(ax, mu, cov, n=50):
    """
    """
    r, v = np.linalg.eig(cov)

    t = np.linspace(0, 2 * np.pi, n)
    xy = np.sqrt(r.reshape((1, 2))) * np.array([np.cos(t), np.sin(t)]).T
    xy = np.dot(xy, v.T)

    ax.plot(mu[0] + xy[:, 0], mu[1] + xy[:, 1])

    '''
    d = np.sqrt(r[0])
    ax.plot(mu[0] + np.array([0, v[0, 0] * d]),
            mu[1] + np.array([0, v[1, 0] * d]),
            label=f'$\\rho$={r[0]:.2g}')
    d = np.sqrt(r[1])
    ax.plot(mu[0] + np.array([0, v[0, 1] * d]),
            mu[1] + np.array([0, v[1, 1] * d]),
            label=f'$\\rho$={r[1]:.2g}')
    ax.legend()
    '''


class ExponentialFit:
    """
    The result of fitting an exponential.

    This can be used as a sequence of parameters, for example::

        a, b, c = expfit.fit1(t, v)

    but also as an object, to access additional methods::

        p = expfit.fit1(t, v)
        lower, upper = p.ci(2)

    Arguments:

    ``x``, ``y``
        The time series.
    ``p``
        The (assumed) optimal parameters.
    ``constraint``
        An optional constraint used in deriving the parameters. Will be used in
        :meth:`ci` if given.

    """
    def __init__(self, x, y, p, constraint=None):
        self._xy = x, y
        self._p = tuple(p)
        self._np = len(self._p)
        self._nt = len(x)
        self._cst = constraint
        self._err = None
        self._cov = None

    def __len__(self):
        return self._np

    def __getitem__(self, subscript):
        return self._p.__getitem__(subscript)

    def __str__(self):
        return ' '.join(f'{i:+.5e}' for i in self._p)

    def ci(self, i, chi2=2.706, max_iter=100, verbose=False):
        """
        Finds and returns a confidence interval for the parameter at index
        ``i`` using a profile likelihood ratio method.

        The method works by:

        1. Setting a threshold MSE as ``(1 + cut-off) * MSE(p_best)``
        2. Fixing the parameter at its original value plus an offset,
           reoptimising, and increasing until the MSE goes above the
           threshold.
        3. Performing bisection search to find the offset at which the
           threshold is crossed.

        The cut-off is set based on the assumption of an additive Normal
        noise term (``data = model + N(0, sigma^2)``), and then rewriting in
        terms of the MSE, leading to::

            cut-off = chi2 / n

        where chi2 is a percentile from a chi-squared distribution with one
        degree of freedom. The default value is ``chi2 = 2.706`` for 95%
        confidence that the true value is in the interval. To obtain a wider
        bound, set e.g. ``chi2 = 3.841`` for 95% confidence. Other values can
        be obtained from tables or e.g. with scipy (if installed)::

            import scipy

            # 90% confidence interval
            chi2 = scipy.stats.chi2.ppf(0.90, 1)

        Arguments:

        ``i``
            The index of the chosen parameter.
        ``chi2``
            The chi-squared distribution value used to determine the cut-off.
            The default value gives a 90% confidence region.
        ``max_iter``
            The maximum iterations for steps 2 and 3.
        ``verbose``
            Set to ``True`` to print debug messages.

        Returns two full parameter sets, corresponding to the lower and upper
        bounds.
        """
        # Create and cache an error
        if self._err is None:
            self._err = expfit.MultiExponentialError(*self._xy)

        # Set cut-off
        cutoff = (1 + chi2 / self._nt) * self._err(self._p)[0]

        def test(value):
            """ Test the given ``value`` has an error below cut-off. """
            # Create a partial parameter array, omitting i
            p_full = np.array(self._p)
            p_full[i] = value

            # Test the constraint, if given
            if self._cst is not None and not self._cst(p_full):
                return False, np.delete(p_full, i)

            # Evaluate the error and compare
            c = expfit.ConstraintWithFixedParameter(self._cst, p_full, i)
            f = expfit.ErrorWithFixedParameter(self._err, p_full, i)
            p = np.delete(p_full, i)
            with np.errstate(all='ignore'):
                r = expfit.fmin(f, p, constraint=c)
            return r.success and r.error < cutoff, r.x

        bounds = []
        for direction in (1, -1):
            # Expand until upper bound found
            d = 1e-6 * np.abs(self._p[i]) * direction
            for j in range(max_iter):
                if not test(self._p[i] + d)[0]:
                    break
                d *= 2
            if j == 0:  # pragma: no cover
                raise RuntimeError('CI method failed: expansion had no'
                                   ' successful optimiser runs.')

            if verbose:  # pragma: no cover
                print(f'Expanded {self._p[i]} to {self._p[i] + d}'
                      f' in {j} iterations')

            # Bisect
            solution = self._p
            a, b = self._p[i], self._p[i] + d
            for j in range(max_iter):
                c = 0.5 * (a + b)
                if np.abs((c - a) / d) < 1e-6:
                    break
                ok, p = test(c)
                if ok:
                    a = c
                    solution = np.insert(p, i, a)
                else:
                    b = c
            bounds.append(solution)

            if verbose:  # pragma: no cover
                print(f'Found {a} in {j} iterations'
                      f' (MSE {self._err(solution)[0]:.3g})')

        return bounds

    def ci_fisher(self, i, perc=1.645):
        """
        Finds and returns a confidence interval for the parameter at index
        ``i`` using a Fisher information method.

            import scipy

            # 90% confidence interval
            perc = scipy.stats.norm.ppf(0.95)

        Arguments:

        ``i``
            The index of the chosen parameter.
        ``perc``
            The Normal percentile point used to determine the interval.
            The default value gives a 90% confidence region.

        Returns a single value ``x``, for an interval of ``mu - x, mu + x``.
        """
        if self._cov is None:
            self.cov()
        return perc * np.sqrt(self._cov[i, i])

    def cov(self):
        """
        Returns a covariance matrix bassed on the Hessian at the obtained
        solution.

        Specifically::

            Cov = (2 * MSE(p_best) / n) * inverse(Hessian(p_best))

        """
        if self._cov is None:
            # Calculate MSE and Hessian
            if self._err is None:
                self._err = expfit.MultiExponentialError(*self._xy)
            mse, jac, hes = self._err(self._p)

            # Calculate covariance matrix and return
            self._cov = np.linalg.inv(hes) * 2 * mse / self._nt

        return self._cov

    def pl(self, i, lo, hi, evals=25):
        """
        Evaluates a profile log-likelihood for the i-th parameter, ranging from
        ``lo`` to ``hi``.

        Arguments:

        ``i``
            The index of the chosen parameter.
        ``lo``
            The minimum value to test for parameter ``i``.
        ``hi``
            The maximum value to test for parameter ``i``.

        Returns a tuple ``(values, errors)`` containing the scanned values and
        the corresponding MSEs.
        """
        if self._err is None:
            self._err = expfit.MultiExponentialError(*self._xy)

        p_full = np.array(self._p)
        values = np.linspace(lo, hi, evals)
        errors = np.zeros(evals)
        for j, val in enumerate(values):
            p_full[i] = val
            c = expfit.ConstraintWithFixedParameter(self._cst, p_full, i)
            f = expfit.ErrorWithFixedParameter(self._err, p_full, i)
            p = np.delete(p_full, i)
            with np.errstate(all='ignore'):
                r = expfit.fmin(f, p, constraint=c)
                errors[j] = r.error
        return values, errors

