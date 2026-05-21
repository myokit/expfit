#
# Single expontial fits
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
import numpy as np

import expfit


def _decaying(p):
    c = p[2::2]
    return np.all(c < 0) and np.all(c[1:] > c[:-1])


def fit_double_decaying(t, v, plot=False, vet=True):
    """
    Fits a double-exponential ``y = a + b * exp(c * x) + d * exp(e * x)``,
    where ``b`` and ``d`` have the same sign, and ``c`` and ``e`` are both
    negative.
    """
    if vet:
        t, v = expfit.vet_series(t, v)

    # Transform to unit square, to avoid overflows etc
    tr = expfit.UnitSquareTransform(t, v)
    x, y = tr.x, tr.y

    # Create initial plot
    if plot:  # pragma: no cover
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(9, 7.5))
        fig.subplots_adjust(0.095, 0.06, 0.995, 0.995, wspace=0.3, hspace=0.5)
        grd = fig.add_gridspec(2, 2, height_ratios=(2, 1))

        ax0 = fig.add_subplot(grd[0, :])
        ax0.set_xlabel('x')
        ax0.set_ylabel('y')
        code, color = ('-', '#92cc92') if len(x) > 10 else ('x-', 'tab:green')
        ax0.plot(x, y, code, color=color, label='Transformed data')

    # Get an initial estimate, for the dominant rate
    at0, bt0, ct0 = expfit.estimate_initial_single(x, y, vet=False)

    # Catch nans etc.
    if ct0 == 0:
        return tr.detransform(at0, bt0, 0, 0, 0)

    # Catch non-decaying
    if ct0 > 0:
        raise RuntimeError(
            'Initial estimate for c > 0, exponential not decaying')

    # Assume dominant rate found, next rate will have smaller magnitude, but
    # bigger multiplier to stay visible
    # Start with 2 times smaller, but increase if the rates converge
    dt0 = bt0
    et0 = ct0
    bt0 *= 0.7
    q0 = np.array((at0, bt0, ct0, dt0, et0), dtype=float)
    for i in range(1, 6):
        q0[4] *= 0.5
        e = expfit.MultiExponentialError(x, y)
        with np.errstate(all='ignore'):
            r = expfit.fmin(e, q0, constraint=_decaying)
            if plot:  # pragma: no cover
                pass #print(r)
        q = r.x
        if q[2] / q[4] - 1 > 1e-3 and r.success:
            break

    # Estimate noise sensitivity
    if False:
        m = 5
        qx = []
        for i in range(m):
            with np.errstate(all='ignore'):
                e = expfit.MultiExponentialError(x[i::m], y[i::m])
                r = expfit.fmin(e, q, constraint=_decaying)
                if r.success:
                    qx.append(r.x)


    # Mess
    ci(x, y, q, 2)


    # Detransform
    p = tr.detransform(q)

    if plot:  # pragma: no cover
        def pstr(p):
            return ' '.join(f'{i:+.5e}' for i in p)

        p0 = tr.detransform(q0)
        lines = [
            f'Transformed Init: {pstr(q0)}',
            f'             Fit: {pstr(q)}',
            f' Real world Init: {pstr(p0)}',
            f'             Fit: {pstr(p)}']
        ax0.text(0.5, -0.30, '\n'.join(lines), transform=ax0.transAxes,
                 ha='center', font='monospace')



        try:
            known = False
            if len(plot) == 5:
                pk = plot
                qk = tr.transform(pk)
                plot = known = True
        except TypeError:
            pass

        f = expfit.exp
        ax0.plot(x, f(x, q0[:3]), '#999', label='Initial')
        if known:
            ax0.plot(x, f(x, qk[:1]), '#bd5900', ls='--', zorder=3,
                     label='Known offset')
            ax0.plot(x, f(x, (0, qk[1], qk[2])), '#1f701f', ls='--', zorder=3,
                     label='Known 1st',)
            ax0.plot(x, f(x, (0, qk[3], qk[4])), '#961b1c', ls='--', zorder=3,
                     label='Known 2nd')
        ax0.plot(
            x, f(x, q[:1]), lw=1, color='tab:orange', label='Fit offset')
        ax0.plot(
            x, f(x, (0, q[1], q[2])), lw=1, color='tab:green', label='Fit 1st')
        ax0.plot(
            x, f(x, (0, q[3], q[4])), lw=1, color='tab:red', label='Fit 2nd')
        ax0.plot(x, f(x, q), lw=1, color='k', label='Fit')

        if False:
            for i, r in enumerate(qx):
                ax0.plot(x, f(x, (0, r[1], r[2])), lw=1, color='tab:green',
                         label=f'Alt 1st {i + 1}')
                ax0.plot(x, f(x, (0, r[3], r[4])), lw=1, color='tab:red',
                         label=f'Alt 2nd {i + 1}')
                ax0.plot(x, f(x, r), ls='--', lw=1, color='k',
                         label=f'Alt {i + 1}')
        ax0.legend(framealpha=1, ncol=2)

        r = y - f(x, q)
        ax2 = fig.add_subplot(grd[1, 0])
        ax2.set_xlabel('x')
        ax2.set_ylabel('Residuals')
        ax2.plot(x, r, label='Residuals fit')
        ax2.legend()

        ax3 = fig.add_subplot(grd[1, 1])
        ax3.set_xlabel('t')
        ax3.set_ylabel('v')
        label = 'Untransformed'
        if known:
            label = f'{label} (tau1={-1 / pk[2]:.3g}, tau2={-1 / pk[4]:.3g})'
        ax3.plot(t, v, code, color=color, label=label)
        ax3.plot(t, f(t, p0[:3]), '-', label=f'Initial (tau={-1 / p0[2]:.3g})')
        ax3.plot(t, f(t, p), '--',
                 label=f'Fit (tau1={-1 / p[2]:.3g}, tau2={-1 / p[4]:.3g})')
        ax3.legend()

    return p


def ci(x, y, q, ifix=0, cutoff=1e-3, max_iter=100):
    e = expfit.MultiExponentialError(x, y)
    cutoff = e(q)[0] * (1 + cutoff)

    def test(value):
        t = np.copy(q)
        t[ifix] = value
        f = expfit.ErrorWithFixedParameter(e, t, ifix)
        t = np.delete(t, ifix)
        with np.errstate(all='ignore'):
            r = expfit.fmin(f, t, constraint=_decaying)
        if not r.success:
            return False
        return r.error < cutoff

    # Expand until upper bound found
    d = -1e-3 * np.abs(q[ifix])
    for i in range(max_iter):
        if not test(q[ifix] + d):
            break
        d *= 2
    print(f'Expanded from {q[ifix]} to {q[ifix] + d} in {i} iterations')

    # Bisect
    a, b = q[ifix], q[ifix] + d
    for i in range(max_iter):
        c = 0.5 * (a + b)
        if np.abs((c - a) / d) < 1e-6:
            break
        if test(c):
            a = c
        else:
            b = c
    print(f'Found {a} in {i} iterations')

    print(q[ifix], a)
