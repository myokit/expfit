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
    gtol = None
    iterations = None
    time = None

    def __str__(self):
        return '\n'.join((
            f'    message: {self.msg}',
            f'    success: {self.success}',
            f'      score: {self.score}',
            f' root score: {np.sqrt(self.score)}',
            f'          x: {self.x}',
            f'       gtol: {self.gtol}',
            f' iterations: {self.iterations}',
            f'       time: {self.time}s',
        ))


def fmin(f, p0, gtol=1e-5, max_iter=500):
    """
    Performs a Levenberg-Marquardt optimisation of ``f`` starting from ``p0``.

    The function ``f`` is expected to return a tuple
    ``(error, jacobian, hessian)``.

    Optimisation stops when the norm of the jacobian is less than ``gtol``.
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

        if False:
            print()
            print(f'p {p}')
            print(f'm {m}')
            print(f'J {j}')
            print(h)
            print(f'cond {np.linalg.cond(h)}')

        # Suggest next point
        ps = p - np.linalg.solve(h + alpha * eye * h, j)
        fs = f(ps[0])

        # Accept and reduce gradient descent factor if improved
        if fs[0] < m:
            alpha *= 0.1
            p = ps
            m, j, h = fs
        elif alpha < 1e12:
            alpha *= 10
    time = timeit.default_timer() - time

    # Create and return result object
    res = OptResult()
    res.x = p[0]
    res.score = m
    res.gtol = np.linalg.norm(j)
    res.iterations = i
    res.time = time
    if err:
        res.msg = err
    elif i == max_iter:
        res.msg = 'Maximum iterations reached'
    else:
        res.success = True
        res.msg = (f'Ran to {m:.7} (gtol {res.gtol:.7})'
                   f' in {time:.5f}s, {i} iterations.')
    return res

