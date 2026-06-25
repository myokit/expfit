#!/usr/bin/env python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import expfit

n = 300
p0 = np.array([2, 2, -0.5])
ra = np.linspace(-4, 8, n)
rb = np.linspace(-4, 8, n)
rc = np.geomspace(-10, -0.125, n)

t = np.linspace(0, 1, 100)
v0 = p0[0] + p0[1] * np.exp(-t / p0[2])
v = v0 + np.random.default_rng(1).normal(0, 0.15, v0.shape)

p = np.array(expfit.fit1(t, v))
#p[1] *= -1

def m(a, b, c):
    return np.sum((a - v + b * np.exp(-t / c))**2) / len(t)

r = [ra, rb, rc]

def surf(x, y):
    surf = np.zeros((len(r[y]), len(r[x])))
    q = np.copy(p)
    for i, yy in enumerate(r[y]):
        q[y] = yy
        for j, xx in enumerate(r[x]):
            q[x] = xx
            # Linear
            #surf[i, j] = m(*q)
            # Log scale
            surf[i, j] = np.log(m(*q))
            # Inverse
            #surf[i, j] = -1 / m(*q)
    return surf

orders = ((1, 2), (0, 2), (1, 0))

print('Calculating', end='')
surfs = []
for x, y in orders:
    surfs.append(surf(x, y))
    print('.', end='', flush=True)
print()


def fig(logb, logc, trunc):
    fig = plt.figure(figsize=(14, 7.5))
    fig.subplots_adjust(0.05, 0.08, 0.98, 0.98)

    ax0 = fig.add_subplot(2, 2, 1)
    ax1 = fig.add_subplot(2, 2, 2)
    ax2 = fig.add_subplot(2, 2, 3)
    ax3 = fig.add_subplot(2, 2, 4)

    log = 'symlog'
    if logb:
        ax0.set_xscale(log)
        ax2.set_xscale(log)
    if logc:
        ax0.set_yscale(log)
        ax1.set_yscale(log)
    for ax in (ax0, ax1, ax2):
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    def do(ax, i):
        x, y = orders[i]
        ax.set_xlabel('abc'[x].replace('c', 'tau'))
        ax.set_ylabel('abc'[y].replace('c', 'tau'))

        surf = surfs[i]
        if trunc:
            cut = 0.1 * np.min(surf)
            surf = np.copy(surf)
            surf[surf < cut] = cut

        ax.contourf(r[x], r[y], -surf, levels=200)
        ax.contour(r[x], r[y], -surf, levels=20, colors='k', linewidths=1)
        ax.axvline(p[x], color='k', lw=0.5)
        ax.axhline(p[y], color='k', lw=0.5)
        ax.axvline(p0[x], color='w', lw=0.5)
        ax.axhline(p0[y], color='w', lw=0.5)

        mn, mx = r[x][0], r[x][-1]
        d = 10**round(np.log10((mx - mn) / 10))
        tlo, thi = d * np.ceil(mn / d), d * np.ceil(mx / d)
        ax.set_xticks(np.arange(tlo, thi, d))

        mn, mx = r[y][0], r[y][-1]
        d = 10**round(np.log10((mx - mn) / 10))
        tlo, thi = d * np.ceil(mn / d), d * np.ceil(mx / d)
        ax.set_yticks(np.arange(tlo, thi, d))

    do(ax0, 0)
    do(ax1, 1)
    do(ax2, 2)

    ax3.plot(t, p[0] + p[1] * np.exp(-t / p[2]), 'k')
    ax3.plot(t, v)
    ax3.plot(t, v0)

    bname = 'log' if logb else 'lin'
    cname = 'log' if logc else 'lin'
    fname = 'tau-scant' if trunc else 'tau-scan'
    fig.savefig(f'/home/michael/{fname}-{bname}-{cname}.png')
    plt.close(fig)

print('Plotting', end='', flush=True)
fig(False, False, False)
print('.', end='', flush=True)
#fig(True, False, False)
#print('.', end='', flush=True)
fig(False, True, False)
print('.', end='', flush=True)
#fig(True, True, False)
if False:
    print('.', end='', flush=True)
    fig(False, False, True)
    print('.', end='', flush=True)
    #fig(True, False, True)
    #print('.', end='', flush=True)
    fig(False, True, True)
    print('.', end='', flush=True)
    #fig(True, True, True)
print('.')

