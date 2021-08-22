import pstats, cProfile
import numpy as np

import pyximport
pyximport.install()

import ccluster.do_calc

def nump2(n, k):
    a = np.ones((k, n-k+1), dtype=np.int16)
    a[0] = np.arange(n-k+1)
    for j in range(1, k):
        reps = (n-k+j) - a[j-1]
        a = np.repeat(a, reps, axis=1)
        ind = np.add.accumulate(reps)
        a[j, ind[:-1]] = 1-reps[1:]
        a[j, 0] = j
        a[j] = np.add.accumulate(a[j])
    return a.T

k = 5
n = 52
# n = 15

dat_y = nump2(n, k)

x = nump2(n, 2)

print(x[0:1])   
cProfile.runctx("do_calc(x, y, id)", {'do_calc':do_calc, 'x': x[0:1], 'y': dat_y, 'id': 0}, {}, "profile.prof")

s = pstats.Stats("profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
