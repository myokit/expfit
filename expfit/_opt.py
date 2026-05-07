#
# Simple optimiser functions for exponential fits
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import timeit

import numpy as np

import expfit


class OptResult:
    message = 'Not run'
    success = False
    x = None
    score = None
    jac = None
    hes = None
    gtol = None
    iterations = None
    evaluations = None
    accepted = None
    time = None

    def __str__(self):
        p = 5
        hes = np.array2string(self.hes, precision=p).splitlines()
        return '\n'.join((
            f'     message: {self.message}',
            f'     success: {self.success}',
            f'  root score: {np.sqrt(self.score)}',
            f'       score: {self.score}',
            f'    jacobian: {np.array2string(self.jac, precision=p)}',
            f'     hessian: {hes[0]}',
            f'              {hes[1]}',
            f'              {hes[2]}',
            f'           x: {np.array2string(self.x, precision=p)}',
            f'        gtol: {self.gtol}',
            f'  iterations: {self.iterations}',
            f' evaluations: {self.evaluations}',
            f'    accepted: {self.accepted}',
            f'        time: {self.time}s',
        ))


def fmin(f, p0, gtol=1e-6, max_iter=200, verbose=False):
    """
    Performs a Levenberg-Marquardt style optimisation of ``f`` starting from
    ``p0``.

    The function ``f`` is expected to return a tuple
    ``(error, jacobian, hessian)``.

    Optimisation stops when the norm of the jacobian is less than ``gtol``
    or when ``max_iter`` iterations have been performed.
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
    alpha = 0.01

    err = False
    m, j, h = f(p[0])
    evaluations = 0
    accepted = 0
    for iterations in range(max_iter):
        if np.linalg.norm(j) < gtol:
            break

        if verbose:  # pragma: no cover
            print(f'Iteration {1 + iterations}')
            print(f'p {p}')
            print(f'm {m}')
            print(f'J {j}')
            print(h)
            print()

        # Suggest next point
        try:
            ps = p - np.linalg.solve(h + alpha * eye * h, j)
        except np.linalg.LinAlgError:  # pragma: no cover
            '''
            # Try Gauss-newton approximation
            try:
                hx = np.outer(j, j)
                ps = p - np.linalg.solve(hx + alpha * eye * hx, j)
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
        if fs[0] < m:
            alpha *= 0.5
            p = ps
            m, j, h = fs
            accepted += 1
        else:
            alpha *= 10
            if alpha > 1e20:  # pragma: no cover
                err = 'Lambda factor grew too large'
                break
    time = timeit.default_timer() - time

    # Create and return result object
    res = OptResult()
    res.x = p[0]
    res.score = m
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

    def __str__(self):
        return (f'mu ({self.mu_x:.3}, {self.mu_y:.3}),'
                f' {self.offset:.3} + {self.slope:.3} x')
