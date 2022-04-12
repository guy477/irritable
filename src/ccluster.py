
import ccluster
import time
import numpy as np
from math import comb
from sklearn.cluster import KMeans, MiniBatchKMeans


if __name__ == '__main__':
    # This is only accurate for k = 5; n = 52. All other combinations should be used solely for testing
    # unless you know what you're doing.
    k = 5
    n = 52

    # Enter the number of threads for you system
    threads = 8

    # Would you like to precompute the centroids? Recommended.
    precompute = True

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

    # # # # pylint: disable=no-member # # # #
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
    # for i in range(comb(n, k)):
    #     if(z[i][0] != z[i][1]):
    #         # print(z[i])
    #         # for j in range(7):
    #         #     print(ranks[z[i][j]%13] + suits[z[i][j]//13], end = ' ')
    #         # print(z[i][7:])
    #         continue
    #     else:
    #         dupe += 1
    dupe = 480200

    #########################################################################################################
    #                Before we cluster, we need to make sure to remove all invalid rows
    #           and calculate the ehs as a floating point value. This is the filtered/preped
    #                           input for the kmeans clustering algorithm
    #       fluff is the exact same thing as no_dupe_river; only with sparcity for hashing purposes
    #########################################################################################################

    # # # # pylint: disable=no-member # # # #
    ccluster.de_dupe(dupe, comb(n, 2), comb(n, 5), new_file = True)
    zND = np.memmap('results/no_dupe_river.npy', mode = 'r', dtype = np.float64, shape = (comb(n, 2) * (comb(n, k) - dupe), 1))
    fluff = np.memmap('results/fluffy.npy', mode = 'r', dtype = np.float64, shape = (comb(n, 2) * (comb(n, k)), 1))

    # c=-1
    # for i in range(z.shape[0] - 1, z.shape[0] - comb(n, k), -1):
    #     if(z[i][0] != z[i][1]):
    #         for j in range(7):
    #             print(ranks[z[i][j]%13] + suits[z[i][j]//13], end = ' ')
    #         print(z[i][7:], end = ' - ')
    #         print(zND[c])
    #         c -= 1



    #########################################################################################################
    #           Memmapped K-Means clustering algorithm. Not yet successful w/full dataset.
    #########################################################################################################

    if(precompute):
        t = time.time()
        centers = ccluster.kmc2(zND, 200)
        np.save('results/cntrs', centers)
        print('Time spend precalculating centers: ' + str((time.time() - t)/60) + ' Minutes')
    else:
        centers = None

    #######
    #
    # I'm not sure if KMeans or MiniBatchKMeans works better...
    #
    #######
    # kmean = KMeans(200, algorithm='full', init = centers, verbose=True, n_init=1).fit_predict(zND)
    # np.save('results/clstrs', kmean)


    centers = np.load('results/cntrs.npy', mmap_mode = 'r')

    k = MiniBatchKMeans(n_clusters = 200, batch_size=int((zND.shape[0] * 200)**.5), tol = 10e-7, max_no_improvement = None, init = centers, verbose=True, n_init=1).fit(zND)

    np.save('results/adjcntrs', k.cluster_centers_)
    np.save('results/lbls', k.labels_)

    adjcntrs = np.load('results/adjcntrs.npy', mmap_mode = 'r')
    lbls = np.load('results/lbls.npy', mmap_mode = 'r')

    #########################################################################################################
    #                This will convert the hashing compatible memory representation of the EHS
    #                   values into the label value. I think this will be useful for turn
    #            calculations; but I can't say for certain.. first time looking at this in months 
    #########################################################################################################
    ccluster.fluff_2_cntrs(comb(52, 2), comb(52, 5), dupe)

    ## WIP DOES NOT WORK (I DONT THINK LMAO; AGAIN LOOKING AT THIS FOR THE FIRST TIME IN MONTHSSSS)
    ccluster.turn_ehs(0, 0, 0, True)

    turn_prob_dist = np.memmap('results/prob_dist.npy', mode = 'r', dtype = np.float32, shape = (comb(n, 2) * ((comb(n, 4) - 40425)), 46))


    #########################################################################################################
    #                                      Flop and Turn EHS Datasets
    #########################################################################################################

    # turn data set will be a C(52, 2) * C(52, 4) np.uint16 dataset. Potential 990*47 overhead w/royalflush on turn: flop will be 990*47*46 - justifying np.int32 for the flop.
    # flop and turn will be calculated based on the river with exact calculations.

