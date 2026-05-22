#
# Confidence intervals: found by fixing the parameter and reoptimising, and
# then changing the fixed parameter until the error goes above some threshold
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import numpy as np

import expfit


def ci(x, y, p, ifix=0, cutoff=5e-2, max_iter=100, constraint=None,
       verbose=False):
    """
    Finds a confidence interval for a parameter.

    The method works by:

    1. Setting a threshold RMSE as ``(1 + cutoff) * rmse(p)``
    2. Fixing the parameter at its original value plus an offset, reoptimising,
       and increasing until the RMSE goes above the threshold.
    3. Performing bisection search to find the offset at which the threshold
       was crossed.

    Arguments:

    ``x``, ``y``
        The time series.
    ``p``
        The initial parameters, assumed to be optimal. These determine the
        number of exponentials used.
    ``ifix``
        The index in ``p`` of the parameter to find confidence intervals for.
    ``cutoff``
        The RMSE threshold is ``(1 + cutoff) * rmse(p)``.
    ``max_iter``
        The maximum iterations for steps 2 and 3.
    ``verbose``
        Set to ``True`` to print debug messages.

    Returns two full parameter sets, corresponding to the lower and upper
    bounds.
    """
    e = expfit.MultiExponentialError(x, y)
    if False:  # pragma: no cover
        # Re-optimise
        r = expfit.fmin(e, p, constraint=constraint)
        p = r.x
    cutoff = e(p)[0] * (1 + cutoff)

    def test(value):
        q_full = np.copy(p)
        q_full[ifix] = value
        q = np.delete(q_full, ifix)
        if not constraint(q_full):
            return False, q
        f = expfit.ErrorWithFixedParameter(e, q_full, ifix)
        with np.errstate(all='ignore'):
            r = expfit.fmin(f, q, constraint=constraint)
        return r.success and r.error < cutoff, r.x

    bounds = []
    for direction in (1, -1):
        # Expand until upper bound found
        d = 1e-6 * np.abs(p[ifix]) * direction
        for i in range(max_iter):
            if not test(p[ifix] + d)[0]:
                break
            d *= 2
        if verbose:  # pragma: no cover
            print(f'Expanded {p[ifix]} to {p[ifix] + d} in {i} iterations')

        # Bisect
        solution = p
        a, b = p[ifix], p[ifix] + d
        for i in range(max_iter):
            c = 0.5 * (a + b)
            if np.abs((c - a) / d) < 1e-6:
                break
            ok, q = test(c)
            if ok:
                a = c
                solution = np.insert(q, ifix, a)
            else:
                b = c
        if verbose:  # pragma: no cover
            print(f'Found {a} in {i} iterations')

        bounds.append(solution)

    return bounds

