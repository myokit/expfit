#!/usr/bin/env python
import numpy as np
import expfit
import expfit._cerr
from timeit import default_timer




x = np.linspace(1, 5, 100)
y = expfit.expc(x, (300, 2, -2))
x, y = expfit.vet_series(x, y)

print(x.shape, y.shape)
print(x.dtype, y.dtype)
print(x.flags)
print(y.flags)

err = expfit.SingleExponentialError(x, y)

p = np.array((299, 2.1, -3.9), dtype=float)
n = 1000
t0 = default_timer()
for i in range(n):
    r0, j0 = err.mse_jac(p)
t0 = default_timer() - t0




j1 = np.zeros(p.shape, dtype=float)
n = 1000
t1 = default_timer()
for i in range(n):
    r1 = expfit._cerr.mse(x, y, p, j1)
t1 = default_timer() - t1

print(r0, t0)
print(r1, t1)
print(j0)
print(j1)
print(t0 / t1)

