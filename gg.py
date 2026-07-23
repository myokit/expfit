#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

m = 0
s = 50
r = np.random.default_rng(1)

ns = np.concatenate((np.arange(2, 50), np.arange(50, 5000, 50)))
s_mean = np.zeros(len(ns))
s_std = np.zeros(len(ns))

for j, n in enumerate(ns):
    stds = [np.std(r.normal(0, s, n)) for i in range(500)]
    s_mean[j] = np.mean(stds)
    s_std[j] = np.std(stds)

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(ns, s_mean)
ax.plot(ns, s_mean - 2 * s_std)
ax.plot(ns, s_mean + 2 * s_std)
ax.plot(ns, s * (1 + 1.96 / np.sqrt(2 * ns)))
ax.plot(ns, s * (1 - 1.96 / np.sqrt(2 * ns)))

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(ns, s_std)
ax.plot(ns, s / np.sqrt(2 * ns))
plt.show()


plt.show()
