import ccluster
import numpy as np
from math import comb
from random import randint
import kmeans


# This is only accurate for k = 5; n = 52. All other combinations should be used solely for testing
# unless you know what you're doing.
k = 5
n = 20

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
ccluster.river_ehs(n, k, threads, new_file = True)



#                      Load a memory mapping of the river scores.

z = np.memmap('results/river.npy', mode = 'r', dtype = np.int16, shape = (comb(n, 2) * comb(n, k), 2 + 5 + 3))



#            Code snippet to continually print a prettified output of all legal hands
#########################################################################################################
ranks = ('2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A')
suits = ('c', 'd', 'h', 's')
for i in range(z.shape[0]):
    if(z[i][0] != z[i][1]):
        for j in range(7):
            print(ranks[z[i][j]%13] + suits[z[i][j]//13], end = ' ')
        print(z[i][7:])
#########################################################################################################

kmeans.test_cykmeans()

cntrs = kmeans.centers(z[...,7:8], 200)
labels = kmeans.assign(z[...,7:8], cntrs)
clusters = kmeans.cluster(z[...,7:8], cntrs)

print(cntrs)
print(clusters)
print(labels)

print(cntrs.shape)
print(len(clusters))
print(labels.shape)