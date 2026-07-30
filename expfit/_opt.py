#
# Simple optimiser functions for exponential fits
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import timeit

import numpy as np


class LMResult:
    """
    A result returned by :meth:`lm`.

    Properties:

    ``success``
        A boolean indicating success.
    ``message``
        A string indicating success or containing an error message.
    ``x``
        The final parameters.
    ``jac``
        The Jacobian of ``x``.
    ``hes``
        The Hessian of ``x``.
    ``error``
        The final error.
    ``gtol``
        The final norm of the Jacobian.
    ``iterations``
        The number of iterations (including rejections).
    ``evaluations``
        The number of evaluations of the error function.
    ``accepted``
        The number of accepted steps. The number of rejected steps is
        ``iterations - accepted``.
    ``time``
        The time taken, in seconds.

    """
    message = 'Not run'
    success = False
    x = None
    error = None
    jac = None
    hes = None
    gtol = None
    iterations = None
    evaluations = None
    accepted = None
    time = None

    def __str__(self):
        p = 5
        x = np.asarray(self.x)
        jac = np.asarray(self.jac)
        hes = np.asarray(self.hes)
        hes = np.array2string(hes, precision=p).splitlines()
        h = f'     hessian: {hes[0]}'
        if len(hes) > 1:
            h += ''.join([f'\n              {h}' for h in hes[1:]])
        return '\n'.join((
            f'     message: {self.message}',
            f'     success: {self.success}',
            f'  root error: {np.sqrt(self.error)}',
            f'       error: {self.error}',
            f'    jacobian: {np.array2string(jac, precision=p)}',
            h,
            f'           x: {np.array2string(x, precision=p)}',
            f'        gtol: {self.gtol}',
            f'  iterations: {self.iterations}',
            f' evaluations: {self.evaluations}',
            f'    accepted: {self.accepted}',
            f'        time: {self.time}s',
        ))


def lm(f, p0, ftol=None, jtol=1e-7, htol=None, max_iter=1000, constraint=None,
       verbose=False, plot=False):
    """
    Performs a Levenberg-Marquardt (LM) style optimisation of ``f`` starting
    from ``p0``.

    At each step, an LM-style step is proposed::

        p* = p - (H + alpha * diag(H))^-1 J

    where ``p*`` is the proposal, ``p`` is the current position, ``J`` and
    ``H`` are the Jacobian and hessian of the current position, and ``alpha``
    is a scaling factor. Unlike typical LM, we use the analytical ``H`` instead
    of approximating it as ``JT J``.

    The step is accepted if ``p*`` has a lower error than ``p``. If a
    constraint is set, the new position should also satisfy this for the step
    to be accepted. The scaling factor is decreased with every successful step
    (converging to a Newton iteration) and increased with every rejection.



    The method halts successfully when the norm of the Jacobian goes below
    ``gtol``.



    Arguments:

    ``f``
        The function to optimise, must return a tuple
        ``(error, jacobian, hessian)``.
    ``p0``
        A starting position. If a constraint is used, this position should
        satisfy it.
    ``gtol``
        The "gradient tolerance" stopping criteria. The optimisation is deemed
        successful when ``np.linalg.norm(jac) < gtol``, where ``jac`` is the
        jacobian of the current position.
    ``max_iter``
        The maximum number of iterations to try.
    ``constraint``
        An optional constraint. New points for which ``constraint(p) != True``
        are rejected.
    ``verbose``
        Set to ``True`` to print status information at every iteration.
    ``plot``
        Optional parameter to create a plot of the routine's progress.

    Returns an :class:`LMResult`.
    """
    time = timeit.default_timer()

    # Initial point: reshape for matrix multiplication
    p = np.asarray(p0)
    n = np.prod(p.shape)
    p = p.reshape((1, n))

    # Check if constraint holds for initial position
    if constraint is not None and not constraint(p[0]):
        # TODO: This could be an exception?
        res = LMResult()
        res.success = False
        res.message = 'Initial position fails constraint'
        return res

    # Initial error, jacobian, hessian, error state and message
    m, j, h = f(p[0])
    evaluations = 1
    accepted = 0
    err, msg = False, None

    # Check error function returns correct dimensions
    if not np.isscalar(m):
        raise ValueError('MSE must be a scalar')
    j = np.asarray(j)
    if j.shape != n:
        raise ValueError(
            'Jacobian must match shape of initial point.'
            f' Got {j.shape}, expecting ({n},)')
    h = np.asarray(h)
    if len(h.shape) != 2 or h.shape != (n, n):
        raise ValueError(
            'Hessian must match shape of initial point.'
            f' Got {h.shape}, expecting ({n}, {n})')

    # Check a stopping criterion is set
    if ftol is None and jtol is None and htol is None:
        raise ValueError('No stopping criterion set')

    # Factor determining balance between gradient descent and Newton. Bigger
    # number means purer Newton method.
    #alpha = 1
    #alpha = 1000    # Cautious start
    alpha = 1e-3  # Brave start

    # Identity matrix used below.
    eye = np.eye(n)

    # Create storage for plot
    if plot is not False:  # pragma: no cover
        # Position, mse, alpha
        log = [[p[0], m, alpha]]

    # Run
    err, msg = False, None
    for iterations in range(max_iter):
        if verbose:  # pragma: no cover
            print(f'Iteration {1 + iterations}')
            print(f'p {p}')
            print(f'm {m}')
            print(f'J {j}')
            print(h)

        # Suggest next point
        try:
            hinv = np.linalg.inv(h + float(alpha) * eye * h)
            ps = p - hinv.dot(j)
            #ps = p - np.linalg.solve(h + float(alpha) * eye * h, j)
        except np.linalg.LinAlgError:  # pragma: no cover
            '''
            # Try Gauss-newton approximation
            try:
                hx = np.outer(j, j)
                ps = p - np.linalg.solve(hx + float(alpha) * eye * hx, j)
            except np.linalg.linalgError:
                fs = [m * 2]
            else:
                h = hx
                fs = f(ps[0])
                evaluations += 1
            '''
            fs = [m * 2]
        else:
            fs = f(ps[0])
            evaluations += 1

        # Accept and reduce gradient descent factor if improved
        ok = fs[0] < m
        if ok and constraint is not None:
            ok = constraint(ps[0])  # Cast back to shape (n, )
            if verbose and not ok:  # pragma: no cover
                print('Constraint failed')
        if ok:
            if verbose:  # pragma: no cover
                print('Accepted')

            #jhj = j.T.dot(np.linalg.inv(h).dot(j))
            #jhj = j.T.dot(np.linalg.inv(h + float(alpha) * eye * h).dot(j))
            #print(np.linalg.norm(j), np.max(np.abs(2 * (ps - p) / (ps + p))), jhj)  # noqa

            improvement = m - fs[0]
            alpha *= 0.5
            p = ps
            m, j, h = fs
            accepted += 1
        else:
            if verbose:  # pragma: no cover
                print(f'Rejected ({fs[0]}, {m})')
            alpha *= 10

        # Update logged information for plot
        if ok and plot is not False:  # pragma: no cover
            log.append([p[0], m, alpha])

        # Stop?
        if ok:
            if ftol is not None and improvement < ftol:
                msg = 'Optimisation successful (ftol)'
                break
            if jtol is not None and np.linalg.norm(j) < jtol:
                msg = 'Optimisation successful (jtol)'
                break
            if htol is not None and j.T.dot(hinv).dot(j) < 1e-20:
                msg = 'Optimisation successful (htol)'
                break
        elif alpha > 1e20:  # pragma: no cover
            err, msg = True, 'Too many successive failed steps'
            break

    if iterations + 1 == max_iter:
        err, msg = True, 'Maximum iterations reached'

    # Create result object
    res = LMResult()
    res.time = timeit.default_timer() - time
    res.x = p[0]
    res.error = m
    res.jac = j
    res.hes = h
    res.gtol = np.linalg.norm(j)
    res.iterations = 1 + iterations
    res.evaluations = evaluations
    res.accepted = accepted
    res.success = False if err else True
    res.message = msg

    # Create plot
    if plot is not False:  # pragma: no cover
        from ._plot import opt_plot
        res.plot = opt_plot(log)
    else:
        res.plot = False

    return res

