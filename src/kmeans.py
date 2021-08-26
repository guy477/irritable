

import cykmeans

import numpy as np


def centers(data, k, iters=20, reps=1, alg='lloyd'):
    # TODO: prefer candidate centers that have exactly k centers
    data = np.require(data, np.float64, 'C')
    best = float('inf')
    best_cntrs = None
    for _ in range(reps):
        cntrs = _init(data, k)
        assigner = _assigner_factory(alg, data, cntrs)
        c = _kmeans(data, k, cntrs, assigner, iters)    # candidate centers
        s = _score(data, c)
        if s < best:
            best = s
            best_cntrs = c
    return best_cntrs


def assign(data, cntrs):
    data = np.require(data, np.float64, 'C')
    d = cykmeans._sq_distances(data, cntrs)
    a = d.argmin(axis=1)
    return a


def cluster(data, cntrs):
    a = assign(data, cntrs)
    c = []
    for i in range(len(cntrs)):
        ai = a == i
        if True not in ai:
            continue
        c.append( data[ai] )
    return c


def _init(data, k):
    n = len(data)
    cntrs = data[np.random.randint(n, size=k)] 
    return cntrs


def _assigner_factory(alg, data, cntrs):
    sub_factory = {
            'lloyd':_lloyd_assigner,
            'elkan':_elkan_assigner,
            }[alg]
    return sub_factory(data, cntrs)


def _lloyd_assigner(data, cntrs):
    #data = np.require(data, np.float64, 'C')
    dists = np.zeros((len(data), len(cntrs)))
    def lloyd(cntrs):
        d = cykmeans._sq_distances(data, cntrs, dists)
        a = d.argmin(axis=1)
        return a
    return lloyd


class _elkan_assigner(object):
    def __init__(self, data, cntrs):
        self.data = np.require(data, np.float64, 'C')
        k = len(cntrs)
        self.lb = cykmeans._distances(data, cntrs)  # lower-bounds
        self.ub = self.lb.min(axis=1)               # upper-bounds
        self.cu = np.zeros(k)                       # center update-distances
        self.cc = np.zeros((k, k))                  # center-to-center distances
        self.ca = self.lb.argmin(axis=1)            # current assignments
        self.pc = cntrs.copy()                      # centers from previous iteration

    def __call__(self, cntrs):
        cykmeans._elkan(cntrs, self.data, self.lb, self.ub, self.cu, self.cc, self.ca, self.pc)
        self.pc = cntrs.copy()
        return self.ca


def _score(data, cntrs):
    data = np.require(data, np.float64, 'C')
    d = cykmeans._sq_distances(data, cntrs)
    return d.min(axis=1).sum()


def _kmeans(data, k, cntrs, assigner, iters=20):
    assert data.dtype == np.float64 and k >= 1 and iters >= 0
    n = float(len(data))
    prev_a = None
    valid = np.zeros(k, dtype=np.int64)
    for _ in range(iters):
        a = assigner(cntrs)
        cykmeans._adjust_centers(data, cntrs, a, valid)
        # stop updating the centers if less than 1% of the points have changed clusters
        changed = np.sum( prev_a != a ) if prev_a is not None else n
        if changed / n < 0.01:
            break
        prev_a = a.copy()
    return cntrs[valid>=1]

