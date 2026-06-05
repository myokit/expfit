#
# Simple optimiser functions for exponential fits
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import timeit

import numpy as np

import expfit


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


def lm(f, p0, gtol=1e-7, max_iter=200, constraint=None, verbose=False):
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

    Returns an :class:`LMResult`.
    """
    # Old:
    # By default, the Hessian is used to guide
    # the update step, but if this leads to an uninvertible matrix, the more
    # common approximation JTJ is used (where JT is the transpose of the
    # Jacobian). Although the Hessian is more exact, it has been suggested this
    # approximation can be more stable, especially far from the true solution.
    time = timeit.default_timer()

    p = np.asarray(p0)
    n = np.prod(p.shape)
    p = p.reshape((1, n))
    eye = np.eye(n)
    alpha = 1000    # Cautious start
    err = False

    m, j, h = f(p[0])
    evaluations = 1
    accepted = 0

    # Check if constraint holds for initial position
    if constraint is not None and not constraint(p[0]):
        err = 'Initial position fails constraint'

    for iterations in range(max_iter):
        if err:
            break
        if np.linalg.norm(j) < gtol:
            break

        if verbose:  # pragma: no cover
            print(f'Iteration {1 + iterations}')
            print(f'p {p}')
            print(f'm {m}')
            print(f'J {j}')
            print(h)

        # Suggest next point
        try:
            ps = p - np.linalg.solve(h + float(alpha) * eye * h, j)
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
            alpha *= 0.5
            p = ps
            m, j, h = fs
            accepted += 1
        else:
            if verbose:  # pragma: no cover
                print(f'Rejected ({fs[0]}, {m})')
            alpha *= 10
            if alpha > 1e20:  # pragma: no cover
                err = 'Lambda factor grew too large'
        if verbose:  # pragma: no cover
            print()

    time = timeit.default_timer() - time

    # Create and return result object
    res = LMResult()
    res.x = p[0]
    res.error = m
    res.jac = j
    res.hes = h
    res.gtol = np.linalg.norm(j)
    res.iterations = 1 + iterations
    res.evaluations = evaluations
    res.accepted = accepted
    res.time = time
    if err:
        res.message = err
    elif iterations + 1 == max_iter:
        res.message = 'Maximum iterations reached'
    else:
        res.success = True
        res.message = 'Optimisation successful'
    return res


class LeastSquaresFit():
    """
    Creates a least squares fit ``(offset, slope)`` where ``y`` is approximated
    by ``offset + slope * x``.

    Properties: ``offset``, ``slope``, ``mu_x``, ``mu_y``.
    """
    def __init__(self, x, y, vet=True):
        if vet:
            x, y = expfit.vet_series(x, y)
        n = len(x)
        if n < 2:
            raise ValueError('At least 2 points are required')

        self.mu_x = np.mean(x)
        self.mu_y = np.mean(y)
        xx = np.sum(x**2) - n * self.mu_x**2
        xy = np.sum(x * y) - n * self.mu_x * self.mu_y
        self.slope = xy / xx
        self.offset = self.mu_y - self.slope * self.mu_x
        self._p = (self.offset, self.slope)

    def __len__(self):
        return 2

    def __getitem__(self, subscript):
        return self._p.__getitem__(subscript)

    def __str__(self):
        return (f'mu ({self.mu_x:.3}, {self.mu_y:.3}),'
                f' {self.offset:.3} + {self.slope:.3} x')

