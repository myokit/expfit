#!/usr/bin/env python3
#
# Tests the jacobian MOVE INTO SINGLE AND DOUBLE ETC
#
import unittest

import numpy as np

import expfit


class TestXXXXXXX(unittest.TestCase):


    def test_mse_jac_single(self):

        n = 100
        a0, b0, c0 = 1, 2, -3
        t = np.linspace(0, 1, n)
        v = a0 + b0 * np.exp(c0 * t)

        a1, b1, c1 = a0 + 1, b0 + 1, c0 + 1
        v1 = a1 + b1 * np.exp(c1 * t)

        mse = np.sum((v - v1)**2) / n
        print(f'Init mse {mse}')

        def mse_jac(t, v, a, b, c):
            m = 1 / n
            e = np.exp(c * t)
            f = a - v + b * e
            ef = e * f
            da = 2 * m * np.sum(f)
            db = 2 * m * np.sum(ef)
            dc = 2 * m * np.sum(ef * t) * b
            return m * np.sum(f * f), da, db, dc

        m1, da, db, dc = mse_jac(t, v, a1, b1, c1)
        print(f'Init mse {m1}')
        print(f'dmse/da {da}')
        print(f'dmse/db {db}')
        print(f'dmse/dc {dc}')

        def mse_fd(t, v, a, b, c, dx=1e-9):
            m = 1 / n
            r1 = np.sum((v - a - b * np.exp(c * t))**2) * m
            r2 = np.sum((v - (a + dx) - b * np.exp(c * t))**2) * m
            da = (r2 - r1) / dx
            r2 = np.sum((v - a - (b + dx) * np.exp(c * t))**2) * m
            db = (r2 - r1) / dx
            r2 = np.sum((v - a - b * np.exp((c + dx) * t))**2) * m
            dc = (r2 - r1) / dx
            return r1, da, db, dc

        m1, da, db, dc = mse_fd(t, v, a1, b1, c1)
        print(f'Init mse {m1}')
        print(f'FD dmse/da {da}')
        print(f'FD dmse/db {db}')
        print(f'FD dmse/dc {dc}')




        #dx = 1e-3
        #a2, b2, c2 = a1 - da * dx, b1 - db * dx, c1 - dc * dx
        #v2 = a2 + b2 * np.exp(c2 * t)
        #m2 = np.sum((v - v2)**2) / n
        #print(f'Second mse {m2} ({m2 - m1})')

        dx = 0.5
        for i in range(100):
            m1, da, db, dc = mse_jac(t, v, a1, b1, c1)
            #m1, da, db, dc = mse_fd(t, v, a1, b1, c1)
            print(i, m1)
            print(f'  {a0: } {a1: 6f} {-da: 6f}')
            print(f'  {b0: } {b1: 6f} {-db: 6f}')
            print(f'  {c0: } {c1: 6f} {-dc: 6f}')






            a1, b1, c1 = a1 - da * dx, b1 - db * dx, c1 - dc * dx
        print(i + 1, np.sum((v - a1 - b1 * np.exp(c1 * t))**2) / n)




if __name__ == '__main__':  # pragma: no cover
    unittest.main()
