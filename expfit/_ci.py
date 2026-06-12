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
        self._mjh = None
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

    def ci_profile(self, i, level=90, max_iter=100, gtol=1e-7, verbose=False):
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
        degree of freedom, obtained from the given confidence level or
        :class:`CLevel` object.

        Arguments:

        ``i``
            The index of the chosen parameter.
        ``level``
            A :class:`CLevel` or an integer setting the confidence level, e.g.
            90 percent.
        ``max_iter``
            The maximum iterations for steps 2 and 3.
        ``gtol``
            The optimiser tolerance to use. See :meth:`expfit.lm`.
        ``verbose``
            Set to ``True`` to print debug messages.

        Returns two full parameter sets, corresponding to the lower and upper
        bounds.
        """
        if self._err is None:
            raise CIUnavailableError()

        # Set cut-off
        if not isinstance(level, CLevel):
            level = CLevel(level)
        e_hat = self._err.mse(self._p)
        cutoff = (1 + level.chi2() / self._nt) * e_hat
        if verbose:  # pragma: no cover
            print(f'Cut off: {cutoff}')

        # Set initial step size based on FIM bounds with same level
        fim = self.ci_fisher(i, level)

        # Set stopping criterion for bisection, based on cut-off
        bisection_tol = 1e-4 * (cutoff - e_hat)

        # Cache last optimiser result, to speed things up
        self._p_cache = np.array(self._p)

        def test(value):
            """ Test the given ``value`` has an error below cut-off. """
            # Create a partial parameter array, omitting i
            self._p_cache[i] = value

            # Test the constraint, if given
            c = None
            if self._cst is not None:
                if not self._cst(self._p_cache):  # pragma: no cover
                    return False, np.delete(self._p_cache, i), np.nan

                # Create a fixed version
                c = expfit.ConstraintWithFixedParameter(
                    self._cst, self._p_cache, i)

            # Evaluate the error and compare
            f = expfit.ErrorWithFixedParameter(self._err, self._p_cache, i)
            p = np.delete(self._p_cache, i)
            with np.errstate(all='ignore'):
                r = expfit.lm(f, p, constraint=c, gtol=gtol)
            if r.success:
                self._p_cache = np.insert(r.x, i, value)
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
            self._p_cache = np.array(self._p)

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
            self._p_cache = np.array(self._p)

            if verbose:  # pragma: no cover
                print(f'Found {a:.5g} in {j} iterations'
                      f' (MSE {self._err.mse(solution):.5g})')

        return bounds

    def ci_fisher(self, i, level=90):
        """
        Finds and returns a confidence interval for the parameter at index
        ``i`` using a Fisher information method.

        Arguments:

        ``i``
            The index of the chosen parameter.
        ``level``
            A :class:`CLevel` or an integer setting the confidence level, e.g.
            90 percent.

        Returns a single value ``x``, for an interval of ``mu - x, mu + x``.
        """
        if self._cov is None:
            self.cov()

        if not isinstance(level, CLevel):
            level = CLevel(level)

        return level.norm() * np.sqrt(self._cov[i, i])

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

            if self._mjh is None:
                self._mjh = self._err(self._p)
            mse, jac, hes = self._mjh
            self._cov = np.linalg.inv(hes) * (2 * mse / self._nt)

        return self._cov

    def error(self):
        """ Returns the error object used to derive this result, if set. """
        if self._err is None:
            raise CIUnavailableError()
        return self._err

    def jac(self):
        """ Returns the Jacobian at the obtained solution. """
        if self._mjh is None:
            self._mjh = self._err(self._p)
        return self._mjh[1]

    def hes(self):
        """ Returns the Hessian at the obtained solution. """
        if self._mjh is None:
            self._mjh = self._err(self._p)
        return self._mjh[2]

    def mse(self):
        """ Returns the MSE at the obtained solution. """
        if self._mjh is None:
            self._mjh = self._err(self._p)
        return self._mjh[0]

    def mse_cutoff(self, level=90):
        """
        Returns the maximum MSE for a given confidence level (assuming
        Normally distruted noise), as used by :meth:`ci_profile`.

        Arguments:

        ``level``
            A :class:`CLevel` or an integer setting the confidence level, e.g.
            90 percent.

        Returns a scalar MSE.
        """
        if self._err is None:
            raise CIUnavailableError()

        if not isinstance(level, CLevel):
            level = CLevel(level)

        return (1 + level.chi2() / self._nt) * self._err.mse(self._p)

    def profile(self, i, lo, hi, evals=25, gtol=1e-7):
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
        ``gtol``
            The optimiser tolerance to use. See :meth:`expfit.lm`.

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
                r = expfit.lm(f, p, constraint=c, gtol=gtol)
                errors[j] = r.error
        return values, errors


class CIUnavailableError(RuntimeError):
    """
    Raised if confidence intervals are requested but the fit result does not
    support this.
    """
    def __init__(self):
        super().__init__('CI methods unavailable for this exponential fit')


class CLevel():
    """
    Provides hard-coded normal and chi-squared percentile point functions for
    use in confidence intervals.

    Example::

        clevel = CLevel(90)
        print(clevel.norm())

    Use :meth:`supported` to see the supported levels. New levels can be added
    with :meth:`add`. If SciPy is installed, these can be obtained with e.g.::

        import scipy
        print(scipy.stats.chi2.ppf(0.95, 1))   % 90%, 1dof
        print(scipy.stats.norm.ppf(0.975, 1))  % 95%

    Arguments:

    ``level``
        A confidence level, in percent.

    """
    _stats = {  # norm, chi2-df1
        99: (2.5758293035489004, 6.6348966010212145),
        95: (1.959963984540054, 3.841458820694124),
        90: (1.6448536269514722, 2.705543454095404),
        75: (1.1503493803760079, 1.3233036969314664),
        50: (0.6744897501960817, 0.454936423119572),
        25: (0.31863936396437514, 0.10153104426762156),
        10: (0.12566134685507416, 0.01579077409343122),
        5: (0.06270677794321385, 0.003932140000019522),
        1: (0.012533469508069276, 0.00015708785790970184),
    }

    def __init__(self, level):
        try:
            self._norm, self._chi2 = self._stats[int(level)]
        except KeyError:
            raise ValueError(f'Confidence level not supported: {level}')

    @classmethod
    def add(cls, level, norm, chi2):
        """
        Add a confidence level.

        Arguments:

        ``level``
            The integer level, in percent.
        ``norm``
            The normal function percent point function.
        ``chi2``
            The 1-degree of freedom chi-squared percent point function.
        """
        cls._stats[level] = (norm, chi2)

    def norm(self):
        """
        Returns the normal percent point function for this level.
        """
        return self._norm

    def chi2(self):
        """
        Returns the 1 degree of freedom chi-squared percent point function for
        this level.
        """
        return self._chi2

