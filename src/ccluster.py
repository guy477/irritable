
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
# Be sure your 'results/' directory has > 100G
# and that you have either 128G ram or 100G+ swap.

#########################################################################################################
#              Unless you're running this program on an excess of 32 threads, using swap
#              memory off an SSD should not be a bottleneck for this part of the program.
#              OS LIMITATIONS MIGHT CAUSE BOTTLENECKS WHEN DUMPING TO AN EXTERNAL SDD/HDD
#              BE SURE TO RUN THE PROGRAM FROM THE SAME DRIVE THAT CONTAINS YOUR EXTENDED
#                     SWAP MEMORY. IF YOU HAVE ENOUGH RAM, DONT WORRY BOUT IT.
#########################################################################################################

# pylint: disable=no-member
ccluster.river_ehs(n, k, threads, new_file = True)


#########################################################################################################
#                         Load a memory mapping of the river scores to view.
#########################################################################################################
z = np.memmap('results/river.npy', mode = 'r', dtype = np.int16, shape = (comb(n, 2) * comb(n, k), 2 + 5 + 3))


#########################################################################################################
#            Code snippet to continually print a prettified output of all legal hands.
#          Also counts the number of invalids. There's def fancy math to get that number
#########################################################################################################
ranks = ('2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A')
suits = ('c', 'd', 'h', 's')
dupe = 0
for i in range(comb(n, k)):
    if(z[i][0] != z[i][1]):
        print(z[i])
        # for j in range(7):
        #     print(ranks[z[i][j]%13] + suits[z[i][j]//13], end = ' ')
        # print(z[i][7:])
        continue
    else:
        dupe += 1

#########################################################################################################
#                Before we cluster, we need to make sure to remove all invalid rows
#           and calculate the ehs as a floating point value. This is the filtered/preped
#                           input for the kmeans clustering algorithm
#########################################################################################################

# pylint: disable=no-member
ccluster.de_dupe(dupe, comb(n, 2), comb(n, 5), new_file = True)
zND = np.memmap('results/no_dupe_river.npy', mode = 'r', dtype = np.float32, shape = (comb(n, 2) * (comb(n, k) - dupe), 1))



# c=0
# for i in range(comb(n, k), comb(n, k)*2):
#     if(z[i][0] != z[i][1]):
#         for j in range(7):
#             print(ranks[z[i][j]%13] + suits[z[i][j]//13], end = ' ')
#         print(z[i][7:], end = ' - ')
#         print(zND[c])
#         c += 1



#########################################################################################################
#           Memmapped K-Means clustering algorithm. Not yet successful w/full dataset.
#########################################################################################################

k = MiniBatchKmeansClustering(n_clusters=200, max_iter=200, batch_size = 1/threads).fit(zND)
lbls = k.get_output()
np.save('results/cntrs', lbls[0])

