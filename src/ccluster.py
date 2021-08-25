
import ccluster
import kmeans
import time

import numpy as np

from math import comb
from random import randint
# from sklearn.cluster import MiniBatchKMeans
from pyemma.coordinates.clustering import MiniBatchKmeansClustering



# This is only accurate for k = 5; n = 52. All other combinations should be used solely for testing
# unless you know what you're doing.
k = 5
n = 52

# Enter the number of threads for you system
threads = 8


# If this is your first time running, make sure new_file is set to true.
# This has only been tested on linux systems.
# Be sure your 'results/' directory has > 70G
# and that you have either 128G ram or 100G+ swap.


# Unless you're running this program on an excess of 32 threads, using swap
# memory off an SSD should not be a bottleneck for this part of the program.
# OS LIMITATIONS MIGHT CAUSE BOTTLENECKS WHEN DUMPING TO AN EXTERNAL SDD/HDD
# BE SURE TO RUN THE PROGRAM FROM THE SAME DRIVE THAT CONTAINS YOUR EXTENDED
# SWAP MEMORY. IF YOU HAVE ENOUGH RAM, DONT WORRY BOUT IT.

#ccluster.river_ehs(n, k, threads, new_file = True)



#                      Load a memory mapping of the river scores.

z = np.memmap('results/river.npy', mode = 'r', dtype = np.int16, shape = (comb(n, 2) * comb(n, k), 2 + 5 + 3))



#            Code snippet to continually print a prettified output of all legal hands
#########################################################################################################
ranks = ('2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A')
suits = ('c', 'd', 'h', 's')
dupe = 0
for i in range(comb(n, k), comb(n, k)*2):
    if(z[i][0] != z[i][1]):
        # for j in range(7):
        #     print(ranks[z[i][j]%13] + suits[z[i][j]//13], end = ' ')
        # print(z[i][7:])
        continue
    else:
        dupe += 1


#########################################################################################################

ccluster.de_dupe(dupe, comb(n, 2), comb(n, 5), new_file = True)
zND = np.memmap('results/no_dupe_river.npy', mode = 'r', dtype = np.float64, shape = (comb(n, 2) * (comb(n, k) - dupe), 1))

# dupe=0
# for i in range(comb(n, k), comb(n, k)*2):
#     if(z[i][0] != z[i][1]):
#         for j in range(7):
#             print(ranks[z[i][j]%13] + suits[z[i][j]//13], end = ' ')
#         print(z[i][7:], end = ' - ')
#         print(zND[dupe])
#         dupe += 1

# for i in range(1000):
#     print(zND[i])
# print(z.shape)
# print(zND.shape)

#                Before we cluster, we need to make sure to remove all invalid rows
#########################################################################################################


# we will also need to modify a kmeans clustering algorithm to use memmapped objects to avoid memory prob

k = MiniBatchKmeansClustering(n_clusters=20, max_iter=200, batch_size = .125).fit(zND)
# k.fit(z[...,8])
t = time.time()
lbls = k.get_output()
np.save(lbls[0])
# print(str((t - time.time())/60)[:4])

# for i in range(len(lbls[0])):
#     print(str(zND[i]) + ' - ' + str(lbls[0][i]))




# cntrs = kmeans.centers(z[...,7:8], 200)
# labels = kmeans.assign(z[...,7:8], cntrs)
# clusters = kmeans.cluster(z[...,7:8], cntrs)

# print(cntrs)
# print(clusters)
# print(labels)
# print(cntrs.shape)
# print(len(clusters))
# print(labels.shape)
#########################################################################################################