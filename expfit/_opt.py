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


def lm(f, p0, gtol=1e-7, max_iter=1000, verbose=False, plot=False):
    """
    Performs a Levenberg-Marquardt (LM) style optimisation of ``f`` starting
    from ``p0``.

    At each step, an LM-style step is proposed::

        p* = p - (H + alpha * diag(H))^-1 J

    where ``p*`` is the proposal, ``p`` is the current position, ``J`` and
    ``H`` are the Jacobian and hessian of the current position, and ``alpha``
    is a scaling factor. Unlike typical LM, we use the analytical ``H`` instead
    of approximating it as ``JT J``.

    The step is accepted if ``p*`` has a lower error than ``p``. The scaling
    factor is decreased with every successful step (converging to a Newton
    iteration) and increased with every rejection.

    The method halts successfully when the norm of the Jacobian goes below
    ``gtol``.

    Arguments:

    ``f``
        The function to optimise, must return a tuple
        ``(error, jacobian, hessian)``.
    ``p0``
        A starting position.
    ``gtol``
        The "gradient tolerance" stopping criteria. The optimisation is deemed
        successful when ``np.linalg.norm(jac) < gtol``, where ``jac`` is the
        jacobian of the current position.
    ``max_iter``
        The maximum number of iterations to try.
    ``verbose``
        Set to ``True`` to print status information at every iteration.
    ``plot``
        Optional parameter to create a plot of the routine's progress.

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

    # Check dimensions
    if not np.isscalar(m):
        raise ValueError('MSE must be a scalar')
    j = np.asarray(j)
    if j.shape != n:
        raise ValueError(
            'Jacobian must match shape of initial point.'
            f' Got {j.shape}, expecting ({n}, )')
    h = np.asarray(h)
    if len(h.shape) != 2 or h.shape != (n, n):
        raise ValueError(
            'Hessian must match shape of initial point.'
            f' Got {h.shape}, expecting ({n}, {n})')

    # Store position etc for plot
    if plot is not False:  # pragma: no cover
        # Position, mse, alpha
        log = [[p[0], m, alpha]]

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

        if ok and plot is not False:  # pragma: no cover
            log.append([p[0], m, alpha])

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
    if err:
        res.message = err
    elif iterations + 1 == max_iter:  # pragma: no cover
        res.message = 'Maximum iterations reached'
    else:
        res.success = True
        res.message = 'Optimisation successful'

    # Create plot
    if plot is not False:  # pragma: no cover
        d = (len(p[0]) - 1) // 2

        if isinstance(plot, tuple):
            fig, axa, axb, axc, axm, axl, diagonals = plot
            for line in diagonals:
                line.remove()
        else:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(11, 7.5))
            fig.subplots_adjust(
                0.075, 0.06, 0.99, 0.95, wspace=0.22, hspace=0.4)
            grid = fig.add_gridspec(3, d - 1, height_ratios=(1, 3, 3))

            grd2 = grid[0, :].subgridspec(1, 3)
            axa = fig.add_subplot(grd2[0])
            axa.set_xlabel('Iterations')
            axa.set_ylabel('a')

            axm = fig.add_subplot(grd2[1])
            axm.set_xlabel('Iterations')
            axm.set_ylabel('MSE')
            axm.set_yscale('log')

            axl = fig.add_subplot(grd2[2])
            axl.set_xlabel('Iterations')
            axl.set_ylabel('Alpha')
            axl.set_yscale('log')

            axb, axc = [], []
            for i in range(d - 1):
                ax = fig.add_subplot(grid[1, i])
                axb.append(ax)
                ax.set_xlabel(f'b{1 + i}')
                ax.set_ylabel(f'b{2 + i}')

                ax = fig.add_subplot(grid[2, i])
                axc.append(ax)
                ax.set_xlabel(f'c{1 + i}')
                ax.set_ylabel(f'c{2 + i}')

        a = [row[0][0] for row in log]
        b = [[row[0][1 + 2 * i] for row in log] for i in range(d)]
        c = [[row[0][2 + 2 * i] for row in log] for i in range(d)]
        m = [row[1] for row in log]
        l = [row[2] for row in log]

        n = len(a)
        import matplotlib
        cmap = matplotlib.colormaps['jet']
        norm = matplotlib.colors.Normalize(0, n)
        cols = [cmap(norm(j)) for j in range(n)]

        axa.plot(a, '-')
        axm.plot(m, '-')
        axl.plot(l, '-')
        for j in range(n):
            axa.plot(j, a[j], 's', color=cols[j])
            axm.plot(j, m[j], 's', color=cols[j])
            axl.plot(j, l[j], 's', color=cols[j])

        for i in range(d - 1):
            axb[i].plot(b[i], b[i + 1], '-')
            axc[i].plot(c[i], c[i + 1], '-')
            for j in range(len(a)):
                axb[i].plot(b[i][j], b[i + 1][j], 's', color=cols[j])
                axc[i].plot(c[i][j], c[i + 1][j], 's', color=cols[j])

        diagonals = []
        for i in range(d - 1):
            x = axc[i].get_xlim()
            diagonals.append(axc[i].plot(x, x, '#ccc', ls='--', lw=1)[0])

        # Pass plot back for repeated optimisation plots
        res.plot = fig, axa, axb, axc, axm, axl, diagonals
    else:
        res.plot = False

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

    def __str__(self):
        return (f'mu ({self.mu_x:.3}, {self.mu_y:.3}),'
                f' {self.offset:.3} + {self.slope:.3} x')

