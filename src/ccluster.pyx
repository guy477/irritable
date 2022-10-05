
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# distutils: language = c++
# emd.pyx

import time
import numpy
import random
import concurrent.futures

cimport numpy
cimport cython

from cython.parallel import prange


from libcpp.map cimport map as mapp

from libc.stdio cimport FILE, fopen, fwrite, fscanf, fclose, fprintf
# cython: profile=True
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance
from sklearn.utils import check_array

ctypedef numpy.uint8_t uint8
ctypedef numpy.uint16_t uint16
ctypedef numpy.int16_t int16
ctypedef numpy.int32_t int32
ctypedef numpy.float32_t float32
ctypedef numpy.float64_t float64
ctypedef numpy.int64_t int64

ctypedef numpy.npy_bool boolean


#################################################################################################
# The below code is taken from eval7 - https://pypi.org/project/eval7/
#################################################################################################


cdef extern from "arrays.h":
    unsigned short N_BITS_TABLE[8192]
    unsigned short STRAIGHT_TABLE[8192]
    unsigned int TOP_FIVE_CARDS_TABLE[8192]
    unsigned short TOP_CARD_TABLE[8192]



cdef int CLUB_OFFSET = 0
cdef int DIAMOND_OFFSET = 13
cdef int HEART_OFFSET = 26
cdef int SPADE_OFFSET = 39

cdef int HANDTYPE_SHIFT = 24 
cdef int TOP_CARD_SHIFT = 16 
cdef int SECOND_CARD_SHIFT = 12 
cdef int THIRD_CARD_SHIFT = 8 
cdef int CARD_WIDTH = 4 
cdef unsigned int TOP_CARD_MASK = 0x000F0000 
cdef unsigned int SECOND_CARD_MASK = 0x0000F000 
cdef unsigned int FIFTH_CARD_MASK = 0x0000000F 

cdef unsigned int HANDTYPE_VALUE_STRAIGHTFLUSH = ((<unsigned int>8) << HANDTYPE_SHIFT)
cdef unsigned int HANDTYPE_VALUE_FOUR_OF_A_KIND = ((<unsigned int>7) << HANDTYPE_SHIFT)
cdef unsigned int HANDTYPE_VALUE_FULLHOUSE = ((<unsigned int>6) << HANDTYPE_SHIFT)
cdef unsigned int HANDTYPE_VALUE_FLUSH = ((<unsigned int>5) << HANDTYPE_SHIFT)
cdef unsigned int HANDTYPE_VALUE_STRAIGHT = ((<unsigned int>4) << HANDTYPE_SHIFT)
cdef unsigned int HANDTYPE_VALUE_TRIPS = ((<unsigned int>3) << HANDTYPE_SHIFT)
cdef unsigned int HANDTYPE_VALUE_TWOPAIR = ((<unsigned int>2) << HANDTYPE_SHIFT)
cdef unsigned int HANDTYPE_VALUE_PAIR = ((<unsigned int>1) << HANDTYPE_SHIFT)
cdef unsigned int HANDTYPE_VALUE_HIGHCARD = ((<unsigned int>0) << HANDTYPE_SHIFT)

#@cython.profile(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef unsigned int cy_evaluate(unsigned long long cards, unsigned int num_cards) nogil:
    """
    7-card evaluation function based on Keith Rule's port of PokerEval.
    Pure Python: 20000 calls in 0.176 seconds (113636 calls/sec)
    Cython: 20000 calls in 0.044 seconds (454545 calls/sec)
    """
    cdef unsigned int retval = 0, four_mask, three_mask, two_mask
    
    cdef unsigned int sc = <unsigned int>((cards >> (CLUB_OFFSET)) & 0x1fffUL)
    cdef unsigned int sd = <unsigned int>((cards >> (DIAMOND_OFFSET)) & 0x1fffUL)
    cdef unsigned int sh = <unsigned int>((cards >> (HEART_OFFSET)) & 0x1fffUL)
    cdef unsigned int ss = <unsigned int>((cards >> (SPADE_OFFSET)) & 0x1fffUL)
    
    cdef unsigned int ranks = sc | sd | sh | ss
    cdef unsigned int n_ranks = N_BITS_TABLE[ranks]
    cdef unsigned int n_dups = <unsigned int>(num_cards - n_ranks)
    
    cdef unsigned int st, t, kickers, second, tc, top
    
    if n_ranks >= 5:
        if N_BITS_TABLE[ss] >= 5:
            if STRAIGHT_TABLE[ss] != 0:
                return HANDTYPE_VALUE_STRAIGHTFLUSH + <unsigned int>(STRAIGHT_TABLE[ss] << TOP_CARD_SHIFT)
            else:
                retval = HANDTYPE_VALUE_FLUSH + TOP_FIVE_CARDS_TABLE[ss]
        elif N_BITS_TABLE[sc] >= 5:
            if STRAIGHT_TABLE[sc] != 0:
                return HANDTYPE_VALUE_STRAIGHTFLUSH + <unsigned int>(STRAIGHT_TABLE[sc] << TOP_CARD_SHIFT)
            else:
                retval = HANDTYPE_VALUE_FLUSH + TOP_FIVE_CARDS_TABLE[sc]
        elif N_BITS_TABLE[sd] >= 5:
            if STRAIGHT_TABLE[sd] != 0:
                return HANDTYPE_VALUE_STRAIGHTFLUSH + <unsigned int>(STRAIGHT_TABLE[sd] << TOP_CARD_SHIFT)
            else:
                retval = HANDTYPE_VALUE_FLUSH + TOP_FIVE_CARDS_TABLE[sd]
        elif N_BITS_TABLE[sh] >= 5:
            if STRAIGHT_TABLE[sh] != 0:
                return HANDTYPE_VALUE_STRAIGHTFLUSH + <unsigned int>(STRAIGHT_TABLE[sh] << TOP_CARD_SHIFT)
            else:
                retval = HANDTYPE_VALUE_FLUSH + TOP_FIVE_CARDS_TABLE[sh]
        else:
            st = STRAIGHT_TABLE[ranks]
            if st != 0:
                retval = HANDTYPE_VALUE_STRAIGHT + (st << TOP_CARD_SHIFT)

        if retval != 0 and n_dups < 3:
            return retval

    if n_dups == 0:
        return HANDTYPE_VALUE_HIGHCARD + TOP_FIVE_CARDS_TABLE[ranks]
    elif n_dups == 1:
        two_mask = ranks ^ (sc ^ sd ^ sh ^ ss)
        retval = <unsigned int>(HANDTYPE_VALUE_PAIR + (TOP_CARD_TABLE[two_mask] << TOP_CARD_SHIFT))
        t = ranks ^ two_mask
        kickers = (TOP_FIVE_CARDS_TABLE[t] >> CARD_WIDTH) & ~FIFTH_CARD_MASK
        retval += kickers
        return retval
    elif n_dups == 2:
        two_mask = ranks ^ (sc ^ sd ^ sh ^ ss)
        if two_mask != 0:
            t = ranks ^ two_mask
            retval = <unsigned int>(HANDTYPE_VALUE_TWOPAIR
                + (TOP_FIVE_CARDS_TABLE[two_mask]
                & (TOP_CARD_MASK | SECOND_CARD_MASK))
                + (TOP_CARD_TABLE[t] << THIRD_CARD_SHIFT))
            return retval
        else:
            three_mask = ((sc & sd) | (sh & ss)) & ((sc & sh) | (sd & ss))
            retval = <unsigned int>(HANDTYPE_VALUE_TRIPS + (TOP_CARD_TABLE[three_mask] << TOP_CARD_SHIFT))
            t = ranks ^ three_mask
            second = TOP_CARD_TABLE[t]
            retval += (second << SECOND_CARD_SHIFT)
            t ^= (1U << <int>second)
            retval += <unsigned int>(TOP_CARD_TABLE[t] << THIRD_CARD_SHIFT)
            return retval
    else:
        four_mask = sh & sd & sc & ss
        if four_mask != 0:
            tc = TOP_CARD_TABLE[four_mask]
            retval = <unsigned int>(HANDTYPE_VALUE_FOUR_OF_A_KIND
                + (tc << TOP_CARD_SHIFT)
                + ((TOP_CARD_TABLE[ranks ^ (1U << <int>tc)]) << SECOND_CARD_SHIFT))
            return retval
        two_mask = ranks ^ (sc ^ sd ^ sh ^ ss)
        if N_BITS_TABLE[two_mask] != n_dups:
            three_mask = ((sc & sd) | (sh & ss)) & ((sc & sh) | (sd & ss))
            retval = HANDTYPE_VALUE_FULLHOUSE
            tc = TOP_CARD_TABLE[three_mask]
            retval += (tc << TOP_CARD_SHIFT)
            t = (two_mask | three_mask) ^ (1U << <int>tc)
            retval += <unsigned int>(TOP_CARD_TABLE[t] << SECOND_CARD_SHIFT)
            return retval
        if retval != 0:
            return retval
        else:
            retval = HANDTYPE_VALUE_TWOPAIR
            top = TOP_CARD_TABLE[two_mask]
            retval += (top << TOP_CARD_SHIFT)
            second = TOP_CARD_TABLE[two_mask ^ (1 << <int>top)]
            retval += (second << SECOND_CARD_SHIFT)
            retval += <unsigned int>((TOP_CARD_TABLE[ranks ^ (1U << <int>top) ^ (1 << <int>second)]) << THIRD_CARD_SHIFT)
            return retval


######################
# https://github.com/obachem/kmc2/blob/master/kmc2.pyx
# 
######################
def kmc2(X, k, chain_length=200, afkmc2=True, random_state=None, weights=None):
    """Cython implementation of k-MC2 and AFK-MC2 seeding
    
    Args:
      X: (n,d)-shaped numpy.ndarray with data points (or scipy CSR matrix)
      k: number of cluster centers
      chain_length: length of the MCMC chain
      afkmc2: Whether to run AFK-MC2 (if True) or vanilla K-MC2 (if False)
      random_state: numpy.random.RandomState instance or integer to be used as seed
      weights: n-sized numpy.ndarray with weights of data points (default: uniform weights)
    Returns:
      (k, d)-shaped numpy.ndarray with cluster centers
    """
    # Local cython variables
    cdef numpy.intp_t j, curr_ind
    cdef double cand_prob, curr_prob
    cdef double[::1] q_cand, p_cand, rand_a

    # Handle input
    X = check_array(X, accept_sparse="csr", dtype=numpy.float64, order="C")    
    sparse = not isinstance(X, numpy.ndarray)
    if weights is None:
        weights = numpy.ones(X.shape[0], dtype=numpy.float64)
    if random_state is None or isinstance(random_state, int):
        random_state = numpy.random.RandomState(random_state)
    if not isinstance(random_state, numpy.random.RandomState):
        raise ValueError("RandomState should either be a numpy.random.RandomState"
                         " instance, None or an integer to be used as seed.")

    # Initialize result
    # centers = numpy.zeros((k, X.shape[1]), numpy.float64, order="C")

    # print(centers)

    centers = numpy.reshape(numpy.linspace(0, 1, num = k*X.shape[1], dtype=numpy.float64), (-1, X.shape[1]))
    
    print(centers)
    # Sample first center and compute proposal
    # rel_row = X[random_state.choice(X.shape[0], p=weights/weights.sum()), :]
    # centers[0, :] = rel_row.todense().flatten() if sparse else rel_row
    
    # print(list(map(numpy.linalg.norm, (X - centers[0:1, :])**2)))
    # print(numpy.reshape(euclidean_distances(X, centers[0:1, :], squared = True), newshape = X.shape[0]))
    # print(numpy.reshape(X, newshape=X.shape[0]))
    if afkmc2:
        
        #di = (numpy.array(list(map(numpy.linalg.norm, (X - centers[0:1, :])))))*weights
        #print(di)
        
        di = numpy.min(euclidean_distances(X, centers[0:1, :], squared=True), axis=1)*weights
        print(di)
        print('euclidean^')
        di = numpy.array(list(map(wasserstein_distance, X, centers[0:1, :] )))
        
        # di = numpy.min(euclidean_distances(X, centers[0:1, :], squared=True), axis=1)*weights
        
        # multiplying by weights seems to crash the program..
        # print(di)
        # di = numpy.reshape(euclidean_distances(X, centers[0:1, :], squared=False), newshape=(X.shape[0]))
        # print(di)
        
        q = di/numpy.sum(di) + weights/numpy.sum(weights)  # Only the potentials
        print(centers[0:1, :])
        print(q)
        print(di)
        print(numpy.sum(q))
        print('^wasserstein: centers/q/di/sum(q)')
    
    else:
        q = numpy.copy(weights)
    # Renormalize the proposal distribution
    q = q / numpy.sum(q)

    for i in range(k-1):
        t1 = time.time()
        # Draw the candidate indices
        cand_ind = random_state.choice(X.shape[0], size=(chain_length), p=q).astype(numpy.intp)
        
        # Extract the proposal probabilities
        q_cand = q[cand_ind]
        
        # Compute pairwise distances
        #dist = euclidean_distances(X[cand_ind, :], centers[0:(i+1), :], squared=True)
        
        dist = (numpy.array(list(map(wasserstein_distance, X[cand_ind], centers[0:(i+1), :]))))
        
        # print(dist)
        # Compute potentials
        # p_cand = numpy.min(dist, axis=1)*weights[cand_ind]
        p_cand = numpy.min(dist)*weights[cand_ind]
        # p_cand = dist[0]*weights[cand_ind]
        
        
        # Compute acceptance probabilities
        rand_a = random_state.random_sample(size=(chain_length))
        with nogil, cython.boundscheck(False), cython.wraparound(False), cython.cdivision(True):
            # Markov chain
            for j in range(q_cand.shape[0]):
                cand_prob = p_cand[j]/q_cand[j]
                if j == 0 or curr_prob == 0.0 or cand_prob/curr_prob > rand_a[j]:
                    # Init new chain             Metropolis-Hastings step
                    curr_ind = j
                    curr_prob = cand_prob
        rel_row = X[cand_ind[curr_ind], :]
        # centers[i+1, :] = rel_row.todense().flatten() if sparse else rel_row
        centers[i+1, :] = rel_row
        t2=time.time()
        t = t2-t1
        print('~' + str(t*(k - 1 - i)//60) + ' minutes until finished. ' + str(100*(i/(k-1)))[:4] + '% done     ', end = '\r')
    return centers



#################################################################################################
# My code below.
#################################################################################################
#
# Taken from stack overflow. Fast list combination generator for C(n, k)
#

def nump2(n, k):
    a = numpy.ones((k, n-k+1), dtype=numpy.int16)
    a[0] = numpy.arange(n-k+1)
    for j in range(1, k):
        reps = (n-k+j) - a[j-1]
        a = numpy.repeat(a, reps, axis=1)
        ind = numpy.add.accumulate(reps)
        a[j, ind[:-1]] = 1-reps[1:]
        a[j, 0] = j
        a[j] = numpy.add.accumulate(a[j])
    return a.T


# 
# See if a card appears twice in the hand/table combo. Can be faster.
# 

# @cython.profile(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef numpy.npy_bool contains_duplicates(int16[:] XX) nogil:
    cdef unsigned int count
    cdef unsigned int length
    cdef unsigned int countt = 0
    length = 7
    for count in range(length):        
        for countt in range(count+1, length):
            if(XX[count] == XX[countt]):
                return True
        
    return False


# 
# See if a card appears twice in the hand/table combo. Can be faster.
# 

# @cython.profile(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef numpy.npy_bool contains_duplicates_turn(int16[:] XX) nogil:
    cdef unsigned int count
    cdef unsigned int length
    cdef unsigned int countt = 0
    length = 6
    for count in range(length):        
        for countt in range(count+1, length):
            if(XX[count] == XX[countt]):
                return True
        
    return False

@cython.boundscheck(False) 
@cython.wraparound(False)
cdef numpy.npy_bool contains_duplicates_flop(int16[:] XX) nogil:
    cdef unsigned int count
    cdef unsigned int length
    cdef unsigned int countt = 0
    length = 5
    for count in range(length):        
        for countt in range(count+1, length):
            if(XX[count] == XX[countt]):
                return True
        
    return False

# 
# Check if the hand/board combo (XX) contains a given card (comp)
# 

# @cython.profile(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef numpy.npy_bool contains(int16[:] XX, int comp) nogil:
    cdef unsigned int count
    cdef unsigned int length = 7
    for count in range(length):
        if(XX[count] == comp):
            return True
        
    return False


@cython.boundscheck(False)
@cython.wraparound(False)
cdef numpy.npy_bool contains_flop(int16[:] XX, int comp) nogil:
    cdef unsigned int count
    cdef unsigned int length = 5
    for count in range(length):
        if(XX[count] == comp):
            return True
        
    return False

@cython.boundscheck(False)
@cython.wraparound(False)
cdef numpy.npy_bool contains_turn(int16[:] XX, int comp) nogil:
    cdef unsigned int count
    cdef unsigned int length = 6
    for count in range(length):
        if(XX[count] == comp):
            return True
        
    return False

# 
# Get the Expected Hand Strength for a given hand/board river combination.
# Returns a 'tally' array for the total number of ties, wins, and losses.
# This is an exact calculation so I leave the result as an integer ratio 
# in order to avoid precision loss later down the line.
#  



# @cython.profile(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef void get_ehs_fast(int16[:] j, int16[:] twl_tiewinloss) nogil:
    
    cdef int T_CARDS = 5
    cdef int N_CARDS = 16

    cdef int16 x[45]
    cdef int16 i, k
    cdef int hero, v, c
    cdef unsigned long long mask = 0
    cdef unsigned int seven = 7
    cdef unsigned long long one = 1
    cdef unsigned long long key

    
    c = 0
    for i in range(N_CARDS):
        if not contains(j, i):
            x[c] = i
            c += 1    

    mask |= one << j[2]
    mask |= one << j[3]
    mask |= one << j[4]
    mask |= one << j[5]
    mask |= one << j[6]

    hero = cy_evaluate(one << j[0] | one << j[1] | mask, seven)

    for i in range(0, N_CARDS - T_CARDS - 3):
        for k in range(i+1, N_CARDS - T_CARDS - 2):

            v = cy_evaluate(one << x[i] | one << x[k] | mask, seven)

            if(hero > v):
                twl_tiewinloss[1] += 1
            elif(v > hero):
                twl_tiewinloss[2] += 1
            else:
                twl_tiewinloss[0] += 1
    





# @cython.boundscheck(False) 
# @cython.wraparound(False)
# cdef void ehs_prop_density():




# 
# All major computation is done in C. Only remaining overhead is encountered in the
# below function. For each of the (legal) C(52, 2) * C(50, 5) combinations that represent all 
# of hero's hand/table combos we make C(45, 2) comparisons with the other legal villian hands.
# The cumulative comparisons done is somewhere between (C(52, 7) * C(45, 2)) and 
# (C(52, 2) * C(52, 5) * C(45, 2)). Most of the current optimizations come in the way of
# memory management (minimizing reads/writes to existing/new locations).
# 
# Will formally calculate another time... ~ it go fast ~ but it can go a good bit faster.
# 



# @cython.profile(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
def do_calc(numpy.int64_t os, int16[:, :] x, int16[:, :] y, int dupes):
    cdef unsigned long long int total = 0
    cdef numpy.int64_t i, j, c
    cdef double t1, t2, t
    cdef numpy.int64_t x_shape = x.shape[0]
    cdef numpy.int64_t y_shape = y.shape[0]

    cd = 0

    # cdef numpy.ndarray[int16, ndim=2] z = numpy.empty((x_shape * y_shape, x.shape[1] + y.shape[1] + 3), dtype=numpy.int16)
    z_memmap = numpy.memmap('results/river.npy', mode = 'r+', dtype = numpy.int16, shape = (x_shape * (y_shape - dupes), 3), offset = os)
    z_f_memmap = numpy.memmap('results/river_f.npy', mode = 'r+', dtype = numpy.float32, shape = (x_shape * (y_shape - dupes), 1), offset = os//3 * 2)
    mp_memmap = numpy.memmap('results/map.npy', mode = 'r+', dtype = numpy.ulonglong, shape = (x_shape * (y_shape - dupes), 1), offset = os//3 * 4)
    
    cdef int16 [:, :] z_view = z_memmap[:]
    cdef numpy.float32_t [:, :] z_f_view = z_f_memmap[:]
    
    cdef unsigned long long [:, :] mp_view = mp_memmap[:]

    cdef numpy.ndarray[int16, ndim=1] oh = numpy.empty(7, dtype=numpy.int16)
    cdef int16 [:] oh_view = oh

    cdef unsigned long long one = 1
    cdef unsigned long long key
    for i in range(x_shape):
        t1=time.time()
        for j in range(y_shape):
            oh_view[:2] = x[i]
            oh_view[2:] = y[j]
            if(not contains_duplicates(oh_view)):
                
                get_ehs_fast(oh_view, z_view[cd][:])
                
                key = (one << oh_view[0]) | (one << oh_view[1]) | (one << oh_view[2]) | (one << oh_view[3]) | (one << oh_view[4]) | (one << oh_view[5]) 
                
                z_f_view[cd] = (z_view[cd][1]+.5*z_view[cd][0]) / (z_view[cd][1]+z_view[cd][0]+z_view[cd][2])
                mp_view[cd] = key

                cd += 1
            
        t2=time.time()
        t = t2-t1
        print('~' + str(t*(x_shape - i)//60) + ' minutes until finished. ' + str(100*(i/x_shape))[:4] + '% done     ', end = '\r')
    mp_memmap.flush()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void insertion_sort_inplace_cython_int16(int16[:] A):
    cdef: 
        int i, j
        int16 key
        int length_A = A.shape[0]

    for j in range(1, length_A):
        key = A[j]
        i = j - 1
        while (i >= 0) & (A[i] > key):
            A[i + 1] = A[i]
            i = i - 1
        A[i + 1] = key




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef prob_dist(numpy.float32_t[:, :] cntrs, int[:] lbs, int64 dupes, boolean turn):

    cdef int T_CARDS
    cdef int N_CARDS
    if turn:
        T_CARDS = 5
        N_CARDS = 16
    else:
        T_CARDS = 4
        N_CARDS = 16
        
    
    cdef int16[:, :] x = nump2(N_CARDS, 2)
    cdef int16[:, :] y = nump2(N_CARDS, T_CARDS)

    y_shape_river = y.shape
    x_shape_river = x.shape

    cdef mapp[unsigned long long, int] mp

    cdef numpy.ndarray[int16, ndim=1] oh = numpy.empty(T_CARDS+1, dtype=numpy.int16)
    cdef numpy.ndarray[int16, ndim=1] oh_z = numpy.empty(T_CARDS, dtype=numpy.int16)
    
    cdef int16[:] oh_view = oh[:]
    cdef int16[:] oh_z_view = oh_z[:]
    cdef int16[:] oh_z_view_tmp = numpy.empty(T_CARDS, dtype=numpy.int16)
    
    cdef int c, i, j, tt, k
    cdef long t


    if turn:
        mp_memmap = numpy.memmap('results/map.npy', mode = 'r+', dtype = numpy.ulonglong, shape = (x.shape[0] * (y.shape[0] - dupes), 1))
    else:
        mp_memmap = numpy.memmap('results/map_turn.npy', mode = 'r+', dtype = numpy.ulonglong, shape = (x.shape[0] * (y.shape[0] - dupes), 1))
    
    cdef unsigned long long [:, :] mp_view = mp_memmap[:]
    cdef unsigned long long [:, :] mp_turn_view
    cdef numpy.float32_t[:, :] mp_z

    
    # semi-pseudo hashmap for each possible public card runout. 
    tmp = 0
    for temp in mp_view:
        mp[temp[0]] = tmp
        # print(temp[0])
        tmp += 1


    if turn:
        dist = numpy.memmap('results/river_f.npy', mode = 'c', dtype = numpy.float32, shape = (x.shape[0] * (y.shape[0] - dupes), 1))
        #mp_z = dist[:]
    else:
        dist = numpy.memmap('results/prob_dist_TURN.npy', mode = 'c', dtype = numpy.float32, shape = (x.shape[0] * (y.shape[0] - dupes), N_CARDS - T_CARDS))
        mp_z = dist[:]


        
    x = nump2(N_CARDS, 2)
    y = nump2(N_CARDS, T_CARDS - 1)

    

    # for each possible remaining card, we will store the subsequent index of the river cluster to which it responds.
    ndupes = 0
    for j in range(y.shape[0]):
        oh_view[:2] = x[0]
        oh_view[2:] = y[j]
        if((turn and contains_duplicates_turn(oh_view)) or (not turn and contains_duplicates_flop(oh_view))):
            ndupes += 1

    print(ndupes)
    # for each possible remaining card, we will store the subsequent index of the river cluster to which it responds.
    if turn:
        mp_turn_memmap = numpy.memmap('results/map_turn.npy', mode = 'w+', dtype = numpy.ulonglong, shape = (x.shape[0] * (y.shape[0] - ndupes), 1))
        mp_turn_memmap.flush()

        mp_turn_view = mp_turn_memmap[:]
        prob_dist = numpy.memmap('results/prob_dist_TURN.npy', mode = 'w+', dtype = numpy.float32, shape = (x.shape[0] * (y.shape[0] - ndupes), N_CARDS - T_CARDS - 1))
    else:
        prob_dist = numpy.memmap('results/prob_dist_FLOP.npy', mode = 'w+', dtype = numpy.float32, shape = (x.shape[0] * (y.shape[0] - ndupes), N_CARDS - T_CARDS - 1))
    
    prob_dist.flush()

    cdef numpy.float32_t[:, :] prob_dist_view = prob_dist[:]
    


    cntrss = numpy.array(cntrs).flatten()
    #distrib = [wasserstein_distance(cntrs, cntrss) for c in range(cntrs.shape[0])]
    #print(distrib)p
    print(cntrss)

    cd = 0
    oh = numpy.empty(T_CARDS+1, dtype=numpy.int16)
    oh_view = oh[:]

    cdef unsigned long long one = 1
    cdef unsigned long long key

    for i in range(x.shape[0]):
        # load current portion of dataset to memory using the offset. 
        # fluff_view[:, :] = numpy.memmap('results/fluffy.npy', mode = 'c', dtype = numpy.float32, shape = (y_shape_river[0], 1), offset = i * y_shape_river[0] * 8)[:]
        t1=time.time()
        for j in range(y.shape[0]):
            oh_view[:2] = x[i]
            oh_view[2:] = y[j]
            if(not ((turn and contains_duplicates_turn(oh_view)) or (not turn and contains_duplicates_flop(oh_view)))):
                oh_z_view[:T_CARDS-1] = oh_view[2:]
                if(turn):
                    mp_turn_view[cd] = (one << oh_view[0]) | (one << oh_view[1]) | (one <<oh_view[2]) | (one <<oh_view[3]) | (one <<oh_view[4])
                    
                c = 0
                # print(cd)
                for k in range(N_CARDS):
                    if(not ((turn and contains_turn(oh_view, k)) or (not turn and contains_flop(oh_view, k)))):
                        ## for each datapoint (float value for winning), find the best center given 
                        
                        oh_z_view_tmp[:] = oh_z_view[:]
                        oh_z_view[T_CARDS - 1] = k


                        insertion_sort_inplace_cython_int16(oh_z_view)
                        


                        if(turn):
                            key = (one << oh_z_view[0])  | (one << oh_z_view[1]) | (one << oh_z_view[2])  | (one << oh_z_view[3])  | (one << oh_z_view[4])  | (one << oh_z_view[5]) 
                        else:
                            key = (one << oh_z_view[0])  | (one << oh_z_view[1]) | (one << oh_z_view[2])  | (one << oh_z_view[3])  | (one << oh_z_view[4]) 
                        # prob_dist[cd][c] = wasserstein_distance(mp_z[key], cntrss)
                        
                        # get the centers for each of the EQUALLY LIKELY next cards 
                        if turn:
                            prob_dist_view[cd][c] = cntrs[lbs[mp[key]]][0]
                        else:
                            #prob_dist[cd][c] = cntrs[lbs[mp[key]]][0]
                            #print(mp_z[mp[key]])
                            #print(cntrs[lbs[mp[key]]])
                            prob_dist_view[cd][c] = wasserstein_distance(mp_z[mp[key]], cntrs[lbs[mp[key]]])
                        #print(dist[mp[key]])
                        #(mp_z[mp[key]][1]+.5*mp_z[mp[key]][0]) / (mp_z[mp[key]][1]+mp_z[mp[key]][0]+mp_z[mp[key]][2])
                        
                        oh_z_view[:] = oh_z_view_tmp[:]
                        c += 1

                cd += 1
        
        t2=time.time()
        t = t2-t1
        print('~' + str(t*(x.shape[0] - i)//60) + ' minutes until finished. ' + str(100*(i/x.shape[0]))[:4] + '% done     ', end = '\r')
    if(turn):
        mp_turn_memmap.flush()
    return ndupes


@cython.boundscheck(False) 
@cython.wraparound(False)
def flop_ehs(n, k, threads, dupes, new_file=False):
    adjcntrs = numpy.load('results/adjcntrs_TURN.npy', mmap_mode = 'r+')
    lbls = numpy.load('results/lbls_TURN.npy', mmap_mode = 'r+')
    
    cdef numpy.float32_t[:, :] cntrs = adjcntrs[:]
    cdef int[:] lbs = lbls[:]


    cdef int16[:, :] x = nump2(n, 2)
    cdef int16[:, :] y = nump2(n, k-1)


    return prob_dist(cntrs, lbs, num_dupes_turn(x, y), False)


@cython.boundscheck(False) 
@cython.wraparound(False)
def turn_ehs(n, k, threads, dupes, new_file=False):
    adjcntrs = numpy.load('results/adjcntrs.npy', mmap_mode = 'r+')
    lbls = numpy.load('results/lbls.npy', mmap_mode = 'r+')
    #print(adjcntrs)
    #print(lbls)
    #print(min(lbls))
    #print(max(lbls))
    
    cdef numpy.float32_t[:, :] cntrs = adjcntrs[:]
    cdef int[:] lbs = lbls[:]

    #print(euclidean_distances(cntrs, cntrs))

    return prob_dist(cntrs, lbs, dupes, True)


@cython.boundscheck(False) 
@cython.wraparound(False)
def num_dupes_flop(int16[:, :] x, int16[:, :] y):
    cdef unsigned long long int total = 0
    cdef numpy.int64_t i, j, c
    cdef double t1, t2, t
    cdef numpy.int64_t x_shape = x.shape[0]
    cdef numpy.int64_t y_shape = y.shape[0]

    cdef mapp[unsigned long long, int] mp

    cd = 0

    cdef numpy.ndarray[int16, ndim=1] oh = numpy.empty(5, dtype=numpy.int16)
    cdef int16 [:] oh_view = oh

    
    for j in range(y_shape):
        oh_view[:2] = x[0][:]
        oh_view[2:] = y[j][:]
        if(contains_duplicates(oh_view)):
            cd += 1
    return cd

@cython.boundscheck(False) 
@cython.wraparound(False)
def num_dupes_turn(int16[:, :] x, int16[:, :] y):
    cdef unsigned long long int total = 0
    cdef numpy.int64_t i, j, c
    cdef double t1, t2, t
    cdef numpy.int64_t x_shape = x.shape[0]
    cdef numpy.int64_t y_shape = y.shape[0]

    cdef mapp[unsigned long long, int] mp

    cd = 0

    cdef numpy.ndarray[int16, ndim=1] oh = numpy.empty(6, dtype=numpy.int16)
    cdef int16 [:] oh_view = oh

    
    for j in range(y_shape):
        oh_view[:2] = x[0]
        oh_view[2:] = y[j]
        if(contains_duplicates(oh_view)):
            cd += 1
    return cd

# @cython.profile(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
def num_dupes(int16[:, :] x, int16[:, :] y):
    cdef unsigned long long int total = 0
    cdef numpy.int64_t i, j, c
    cdef double t1, t2, t
    cdef numpy.int64_t x_shape = x.shape[0]
    cdef numpy.int64_t y_shape = y.shape[0]

    cdef mapp[unsigned long long, int] mp

    cd = 0

    cdef numpy.ndarray[int16, ndim=1] oh = numpy.empty(7, dtype=numpy.int16)
    cdef int16 [:] oh_view = oh

    
    for j in range(y_shape):
        oh_view[:2] = x[0]
        oh_view[2:] = y[j]
        if(contains_duplicates(oh_view)):
            cd += 1
    return cd

@cython.boundscheck(False) 
@cython.wraparound(False)
def river_ehs(n, k, threads, new_file=False):
    x = nump2(n, 2)
    y = nump2(n, k)

    dupes = num_dupes(x, y)
    
    print(dupes)
    if(new_file):
        z = numpy.memmap('results/river.npy', mode = 'w+', dtype = numpy.int16, shape = ((y.shape[0] - dupes) * x.shape[0], 3))
        z_f = numpy.memmap('results/river_f.npy', mode = 'w+', dtype = numpy.float32, shape = ((y.shape[0] - dupes) * x.shape[0], 1))
        mp = numpy.memmap('results/map.npy', mode = 'w+', dtype = numpy.ulonglong, shape = ((y.shape[0] - dupes) * x.shape[0], 1))
            
        z.flush()
        z_f.flush()
        mp.flush()

    chunksize = len(x) // (threads-1)
    print('Starting river EHS')    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # change threads to appropriate number of workers for your system
        futures = []
        for i in range(threads-1):
            strt = i * chunksize
            stp = ((i + 1) * chunksize) if i != (threads - 2) else len(x)
            print(strt, end = ' ')
            print(stp)
            futures.append(executor.submit(do_calc, strt * (y.shape[0]-dupes) * 3 * 2, x[strt:stp], y, dupes))
        concurrent.futures.wait(futures)

        output = [f.result() for f in futures]
    
    return dupes

    