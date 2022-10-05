
import ccluster
import kmeans
import time

from scipy.stats import wasserstein_distance

import numpy as np
import pandas as pd

from math import comb
from random import randint
from sklearn.cluster import MiniBatchKMeans, KMeans
# from pyemma.coordinates.clustering import KmeansClustering

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



# This is only accurate for k = 5; n = 52. All other combinations should be used solely for testing
# unless you know what you're doing.
k = 5
n = 16

# Enter the number of threads for you system
threads = 16




# If this is your first time running, make sure new_file is set to true.
# This has only been tested on linux systems.
# Be sure your 'results/' directory has > 70G
# and that you have either 128G ram or 100G+ swap.


# Unless you're running this program on an excess of 32 threads, using swap
# memory off an SSD should not be a bottleneck for this part of the program.
# OS LIMITATIONS MIGHT CAUSE BOTTLENECKS WHEN DUMPING TO AN EXTERNAL SDD/HDD
# BE SURE TO RUN THE PROGRAM FROM THE SAME DRIVE THAT CONTAINS YOUR EXTENDED
# SWAP MEMORY. IF YOU HAVE ENOUGH RAM, DONT WORRY BOUT IT.


dupes = ccluster.river_ehs(n, k, threads, new_file = True)



#                      Load a memory mapping of the river scores.

# z = np.memmap('results/river.npy', mode = 'r', dtype = np.int16, shape = (comb(n, 2) * (comb(n, k) - dupes), 3))
z = np.memmap('results/river_f.npy', mode = 'r', dtype = np.float32, shape = (comb(n, 2) * (comb(n, k) - dupes), 1))

print(z)

# #########################################################################################################


# we will also need to modify a kmeans clustering algorithm to use memmapped objects to avoid memory prob

# k = MiniBatchKmeansClustering(n_clusters=200, max_iter=200, batch_size = .5).fit(np.asarray(zND))
# k = MiniBatchKMeans(n_clusters=200, max_iter=200, )

# highly recommended.

precompute = True

if precompute:
    t = time.time()
    
    print('precomputing centers')
    centers = ccluster.kmc2(z, 10, chain_length=50, afkmc2=True)
    np.save('results/cntrs', centers)

    print('Time spent precalculating centers: ' + str((time.time() - t)/60) + 'm')
else:
    centers = None

centers = np.load('results/cntrs.npy', mmap_mode = 'r')

# miniK = MiniBatchKMeans(n_clusters = 200, batch_size=int((z.shape[0] * 200)**.5), tol = 10e-6, max_no_improvement = None, init = centers, verbose=True, n_init=1).fit(z)
miniK = MiniBatchKMeans(n_clusters = 10, tol = 10e-6, max_no_improvement = None, verbose=True, n_init=1).fit(z)

np.save('results/adjcntrs', miniK.cluster_centers_)
np.save('results/lbls', miniK.labels_)



adjcntrs = np.load('results/adjcntrs.npy', mmap_mode = 'r')
lbls = np.load('results/lbls.npy', mmap_mode = 'r')

unique, counts = np.unique(adjcntrs, return_counts=True)

print(len(counts))
print(sorted(centers))

dupes = ccluster.turn_ehs(0, 0, 0, dupes, True)


turn_prob_dist = np.memmap('results/prob_dist_TURN.npy', mode = 'r', dtype = np.float32, shape = (comb(n, 2) * ((comb(n, 4) - dupes)), n - k - 1))

# we will now like to refluff the labels to allow for index mappings

pd.DataFrame(turn_prob_dist)[:(comb(n, 4) - dupes)].to_excel('turn_prob_dist_2c3c.xlsx')

# pd.DataFrame(turn_prob_dist)[(comb(n, 4) - dupes)*7:(comb(n, 4) - dupes)*8].to_excel('turn_prob_dist_2c3c.xlsx')


#########################################################################################################
#########################################################################################################
# dupe = 0
# ranks = ('2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A')
# suits = ('c', 'd', 'h', 's')
# for i in range(comb(n, 4)):
#     if(not x[0][0] in y[i] and not x[0][1] in y[i]):
#         print(ranks[x[0][0]%13] + suits[x[0][0]//13], end = ' ')
#         print(ranks[x[0][1]%13] + suits[x[0][1]//13], end = '_')
#         print(ranks[y[i][0]%13] + suits[y[i][0]//13], end = ' ')
#         print(ranks[y[i][1]%13] + suits[y[i][1]//13], end = ' ')
#         print(ranks[y[i][2]%13] + suits[y[i][2]//13], end = ' ')
#         print(ranks[y[i][3]%13] + suits[y[i][3]//13], end = ' - ')
#         print(turn_prob_dist[dupe])
#         dupe += 1
#########################################################################################################
#########################################################################################################


#########################################################################################################

if precompute:
    t = time.time()
    
    print('precomputing centers ---- TURN')
    centers_TURN = ccluster.kmc2(turn_prob_dist, 10, chain_length=50, afkmc2=True)
    np.save('results/cntrs_TURN', centers_TURN)

    print('Time spent precalculating centers: ' + str((time.time() - t)/60) + 'm')
else:
    centers_TURN = None

centers_TURN = np.load('results/cntrs_TURN.npy', mmap_mode = 'r')

# miniK = MiniBatchKMeans(n_clusters = 100, batch_size=int((turn_prob_dist.shape[0] * 200)**.5), tol = 10e-6, max_no_improvement = None, init = centers_TURN, verbose=True, n_init=1).fit(turn_prob_dist)
miniK = MiniBatchKMeans(n_clusters = 10, tol = 10e-6, max_no_improvement = None, verbose=True, n_init=1).fit(turn_prob_dist)

np.save('results/adjcntrs_TURN', miniK.cluster_centers_)
np.save('results/lbls_TURN', miniK.labels_)

adjcntrs = np.load('results/adjcntrs_TURN.npy', mmap_mode = 'r')
lbls = np.load('results/lbls_TURN.npy', mmap_mode = 'r')

unique, counts = np.unique(adjcntrs, return_counts=True)

print(centers_TURN)

dupes = ccluster.flop_ehs(n, k, 0, True)
print(adjcntrs)
print(lbls)

flop_prob_dist = np.memmap('results/prob_dist_FLOP.npy', mode = 'r', dtype = np.float16, shape = (comb(n, 2) * ((comb(n, 3) - dupes)), n - k - 2))
pd.DataFrame(flop_prob_dist).to_excel('flop_prob_dist_ALL.xlsx')