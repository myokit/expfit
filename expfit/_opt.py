#
# Simple optimiser for exponential fits
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import timeit

import numpy as np


class OptResult:
    msg = 'Not run'
    success = False
    x = None
    score = None
    jac = None
    hes = None
    gtol = None
    iterations = None
    time = None

    def __str__(self):
        p = 5
        hes = np.array2string(self.hes, precision=p).splitlines()
        return '\n'.join((
            f'    message: {self.msg}',
            f'    success: {self.success}',
            f' root score: {np.sqrt(self.score)}',
            f'      score: {self.score}',
            f'   jacobian: {np.array2string(self.jac, precision=p)}',
            f'    hessian: {hes[0]}',
            f'             {hes[1]}',
            f'             {hes[2]}',
            f'          x: {np.array2string(self.x, precision=p)}',
            f'       gtol: {self.gtol}',
            f' iterations: {self.iterations}',
            f'       time: {self.time}s',
        ))


def fmin(f, p0, gtol=1e-6, max_iter=50, verbose=False):
    """
    Performs a Levenberg-Marquardt optimisation of ``f`` starting from ``p0``.

    The function ``f`` is expected to return a tuple
    ``(error, jacobian, hessian)``.

    Optimisation stops when the norm of the jacobian is less than ``gtol``,
    when ``max_iter`` iterations have been performed, or when the problem is
    ill-conditioned
    """
    time = timeit.default_timer()

    p = np.asarray(p0)
    n = np.prod(p.shape)
    p = p.reshape((1, n))
    eye = np.eye(n)
    alpha = 0.01

    err = False
    m, j, h = f(p[0])
    for i in range(max_iter):
        if np.linalg.norm(j) < gtol:
            break
        if np.linalg.cond(h) > 1e15:
            err = 'Ill-conditioned Hessian'
            break

        if verbose:  # pragma: no cover
            print(f'Iteration {1 + i}')
            print(f'p {p}')
            print(f'm {m}')
            print(f'J {j}')
            print(h)
            print(f'cond {np.linalg.cond(h)}')
            print()

        # Suggest next point
        ps = p - np.linalg.solve(h + alpha * eye * h, j)
        fs = f(ps[0])

        # Accept and reduce gradient descent factor if improved
        if fs[0] < m:
            alpha *= 0.1
            p = ps
            m, j, h = fs
        else:
            alpha *= 10
            if alpha > 1e100:  # pragma: no cover
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
    res.iterations = i
    res.time = time
    if err:
        res.msg = err
    elif i == max_iter:
        res.msg = 'Maximum iterations reached'
    else:
        res.success = True
        res.msg = 'Optimisation successful'
    return res

