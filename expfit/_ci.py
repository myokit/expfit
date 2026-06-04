#
# Fitting result with confidence interval methods
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#

import numpy as np

import expfit


class ExponentialFit:
    """
    The result of fitting an exponential.

    This can be used as a sequence of parameters, for example::

        a, b, c = expfit.fit1(t, v)

    For succesful fits, additional methods are available to calculate
    confidence intervals.

        p = expfit.fit1(t, v)
        lower, upper = p.ci_fisher(2)

    If CI is unavailable, these methods will raise a
    :class:`CIUnavailableError`. To check availability, :meth:`ci_available()`
    can be used.

    Arguments:

    ``x``, ``y``
        The time series.
    ``p``
        The (assumed) optimal parameters.
    ``error``
        An optional error object, used in CI methods.
    ``constraint``
        An optional constraint object, used CI methods.

    """
    def __init__(self, x, y, p, error=None, constraint=None):
        self._xy = x, y
        self._p = tuple(p)
        self._np = len(self._p)
        self._nt = len(x)

        self._err = error
        self._cst = constraint
        self._cov = None

    def __len__(self):
        return self._np

    def __getitem__(self, subscript):
        return self._p.__getitem__(subscript)

    def __str__(self):
        return ' '.join(f'{i:+.5e}' for i in self._p)

    def ci_available(self):
        """
        Returns ``True`` only if CI methods are available for this result.
        """
        return self._err is not None

    def ci_profile(self, i, chi2=2.706, max_iter=100, verbose=False):
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
        if self._err is None:
            raise CIUnavailableError()

        # Set cut-off
        e_hat = self._err(self._p)[0]
        cutoff = (1 + chi2 / self._nt) * e_hat
        if verbose:  # pragma: no cover
            print(f'Cut off: {cutoff}')

        # Set initial step size based on 90% FIM bounds
        # TODO Add some class for CILevel that has these baked in so we can use
        # the appropriate level
        fim = self.ci_fisher(i)

        # Set stopping criterion for bisection, based on cut-off
        bisection_tol = 1e-4 * (cutoff - e_hat)

        def test(value):
            """ Test the given ``value`` has an error below cut-off. """
            # Create a partial parameter array, omitting i
            p_full = np.array(self._p)
            p_full[i] = value

            # Test the constraint, if given
            c = None
            if self._cst is not None:
                if not self._cst(p_full):  # pragma: no cover
                    return False, np.delete(p_full, i)

                # Create a fixed version
                c = expfit.ConstraintWithFixedParameter(self._cst, p_full, i)

            # Evaluate the error and compare
            f = expfit.ErrorWithFixedParameter(self._err, p_full, i)
            p = np.delete(p_full, i)
            with np.errstate(all='ignore'):
                r = expfit.lm(f, p, constraint=c, verbose=False)

            return r.error < cutoff, r.x, r.error

        # Find the boundaries
        bounds = []
        for direction in (-1, 1):

            # Start from the FIM bounds, expanding if necessary
            d = fim * direction
            dd = fim * 0.1 * direction
            for j in range(max_iter):
                if not test(self._p[i] + d)[0]:
                    break
                d += dd
                dd *= 2
            if j + 1 == max_iter:  # pragma: no cover
                raise RuntimeError(
                    'Unable to find upper/lower limit for profile CI:'
                    ' maximum iterations reached')

            # Bisect
            solution = self._p
            a, b = self._p[i], self._p[i] + d
            e_old = e_hat
            for j in range(max_iter):
                c = 0.5 * (a + b)
                ok, p, e_new = test(c)
                if np.abs(e_new - e_old) < bisection_tol:
                    break
                e_old = e_new
                if ok:
                    a = c
                    solution = np.insert(p, i, a)
                else:
                    b = c
            bounds.append(solution)

            if verbose:  # pragma: no cover
                print(f'Found {a:.5g} in {j} iterations'
                      f' (MSE {self._err(solution)[0]:.5g})')

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

        This is equivalent to the inverse Fisher information matrix, under the
        assumption of an added Gaussian noise term.
        """
        if self._cov is None:
            if self._err is None:
                raise CIUnavailableError()

            mse, jac, hes = self._err(self._p)
            self._cov = np.linalg.inv(hes) * 2 * mse / self._nt

        return self._cov

    def profile(self, i, lo, hi, evals=25):
        """
        Profiles the MSE for the i-th parameter, ranging from ``lo`` to ``hi``.

        For each value, the optimisation is re-run, keeping the i-th parameter
        fixed.

        Arguments:

        ``i``
            The index of the chosen parameter.
        ``lo``
            The minimum value to test for parameter ``i``.
        ``hi``
            The maximum value to test for parameter ``i``.

        Returns a tuple ``(values, errors)`` containing the tested parameter
        values and their MSEs.
        """
        if self._err is None:
            raise CIUnavailableError()

        p_full = np.array(self._p)
        values = np.linspace(lo, hi, evals)
        errors = np.zeros(evals)
        for j, val in enumerate(values):
            p_full[i] = val
            c = None
            if self._cst is not None:  # pragma: no cover
                c = expfit.ConstraintWithFixedParameter(self._cst, p_full, i)
            f = expfit.ErrorWithFixedParameter(self._err, p_full, i)
            p = np.delete(p_full, i)
            with np.errstate(all='ignore'):
                r = expfit.lm(f, p, constraint=c)
                errors[j] = r.error
        return values, errors


class CIUnavailableError(RuntimeError):
    def __init__(self):
        super().__init__('CI methods unavailable for this exponential fit')

