#!/usr/bin/env python
import numpy as np
import expfit
import expfit._cerr
from timeit import default_timer




x = np.linspace(1, 5, 10)
y = expfit.expc(x, (300, 2, -2))
x, y = expfit.vet_series(x, y)

print(x.shape, y.shape)
print(x.dtype, y.dtype)
print(x.flags)
print(y.flags)

err = expfit.SingleExponentialError(x, y)

n = 1000
t0 = default_timer()
for i in range(n):
    r0 = err.mse((299, 2.1, -1.9))
t0 = default_timer() - t0


n = 1000
t1 = default_timer()
for i in range(n):
    r1 = expfit._cerr.mse(x, y, 299, 2.1, -1.9)
t1 = default_timer() - t1

print(r0, t0)
print(r1, t1)
print(t0 / t1)

