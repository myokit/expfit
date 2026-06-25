#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import expfit

def fig(logb=False, logc=False, n=100, lo=0.1, hi=4):
    fig = plt.figure(figsize=(14, 8))
    fig.subplots_adjust(0.05, 0.05, 0.98, 0.98)

    ax0 = fig.add_subplot(2, 2, 1)
    ax1 = fig.add_subplot(2, 2, 2)
    ax2 = fig.add_subplot(2, 2, 3)
    ax3 = fig.add_subplot(2, 2, 4)

    a0 = 2
    b0 = np.log(2) if logb else 2
    c0 = np.log(2) if logc else 2

    x = np.linspace(0, 1, 100)
    if logb and logc:
        y0 = a0 + np.exp(b0) * np.exp(np.exp(c0) * x)
        def m(a, b, c):
            return np.sum((a - y + np.exp(b) * np.exp(np.exp(c) * x))**2) / len(x)
    elif logb:
        y0 = a0 + np.exp(b0) * np.exp(c0 * x)
        def m(a, b, c):
            return np.sum((a - y + np.exp(b) * np.exp(c * x))**2) / len(x)
    elif logc:
        y0 = a0 + b0 * np.exp(np.exp(c0) * x)
        def m(a, b, c):
            return np.sum((a - y + b * np.exp(np.exp(c) * x))**2) / len(x)
    else:
        y0 = a0 + b0 * np.exp(c0 * x)
        def m(a, b, c):
            return np.sum((a - y + b * np.exp(c * x))**2) / len(x)
    rng = np.random.default_rng(1)
    y = y0 + rng.normal(0, 0.15, y0.shape)

    p = expfit.fit1(x, y)
    print(p)

    ax3.plot(x, y)
    ax3.plot(x, y0)

    ra = np.linspace(lo, hi, n)
    rb = np.linspace(np.log(lo), np.log(hi), n) if logb else np.linspace(lo, hi, n)
    rc = np.linspace(np.log(lo), np.log(hi), n) if logc else np.linspace(lo, hi, n)

    def do(ax, ri, rj, li, lj, f):
        ax.set_xlabel(li)
        ax.set_ylabel(lj)
        surf = np.zeros((len(ri), len(rj)))
        for i, a in enumerate(ri):
            for j, b in enumerate(rb):
                surf[i, j] = np.log(f(a, b))
        ax.contourf(ri, rj, surf)

    la = 'a'
    lb = 'log(b)' if logb else 'b'
    lc = 'log(c)' if logc else 'c'
    do(ax0, rb, rc, lb, lc, lambda b, c: m(a0, b, c))
    do(ax1, ra, rc, la, lc, lambda a, c: m(a, b0, c))
    do(ax2, rb, ra, lb, la, lambda b, a: m(a, b, c0))

    import matplotlib.ticker as ticker
    ff = ticker.FuncFormatter(lambda x, _: f'{np.exp(x):.3}')
    xt = ax1.get_xticks()
    xt = np.log(xt[xt > 0])
    if logb:
        ax0.set_xticks(xt)
        ax0.xaxis.set_major_formatter(ff)
        ax2.set_xticks(xt)
        ax2.xaxis.set_major_formatter(ff)
    if logc:
        ax0.set_yticks(xt)
        ax0.yaxis.set_major_formatter(ff)
        ax1.set_yticks(xt)
        ax1.yaxis.set_major_formatter(ff)

    ax0.axvline(b0, color='w', lw=0.5)
    ax0.axhline(c0, color='w', lw=0.5)
    ax0.axvline(p[1], color='w', lw=0.5)
    ax0.axhline(p[2], color='w', lw=0.5)


    ax1.axvline(p[0], color='w', lw=0.5)
    ax1.axhline(p[2], color='w', lw=0.5)

    ax2.axvline(p[1], color='w', lw=0.5)
    ax2.axhline(p[0], color='w', lw=0.5)

    ax4.plot(x, p[0] + p[1] * np.exp(p[2] * x), 'k')


lo = 0.01
hi = 10
n = 100
fig(False, False, n, lo, hi)
#fig(True, False, n, lo, hi)
#fig(False, True, n, lo, hi)
#fig(True, True, n, lo, hi)


plt.show()
