
#!/usr/bin/env python3
#cython: language_level=3
# -*- coding: utf-8 -*-
# distutils: language = c++


import time

import random
import concurrent.futures
import concurrent.futures

import numpy
cimport numpy
cimport cython
numpy.import_array()

from cython.parallel import prange

from libcpp.map cimport map as mapp

from libc.stdlib cimport calloc, free
from libc.stdio cimport FILE, fopen, fwrite, fscanf, fclose, fprintf
# cython: profile=True
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist, squareform

from sklearn.utils import check_array

"""
Wraper for C EMD implementation
"""
from scipy.spatial.distance import cdist



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



#####################
#####################


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
    
    cdef float32[:, :]XX = numpy.empty((len(X), len(X[0])), dtype = numpy.float32)

    #XX[:] = X[:]
        
    sparse = False
    if weights is None:
        weights = numpy.ones(XX.shape[0], dtype=numpy.float32)
    if random_state is None or isinstance(random_state, int):
        random_state = numpy.random.RandomState(random_state)
    if not isinstance(random_state, numpy.random.RandomState):
        raise ValueError("RandomState should either be a numpy.random.RandomState"
                         " instance, None or an integer to be used as seed.")

    # Initialize result

    cntrs = numpy.zeros((k, X.shape[1]), dtype=numpy.float32)

    # print(centers)

    #centers = numpy.reshape(numpy.linspace(0, 1, num = k*X.shape[1], dtype=float64), (-1, X.shape[1]))
    
    #print(centers)

    # Sample first center and compute proposal

    print(X[random_state.choice(X.shape[0], p=weights/weights.sum())])
    cntrs[0] = X[random_state.choice(X.shape[0], p=weights/weights.sum())]

    
    
    # print(list(map(numpy.linalg.norm, (X - centers[0:1, :])**2)))
    # print(numpy.reshape(euclidean_distances(X, centers[0:1, :], squared = True), newshape = X.shape[0]))
    # print(numpy.reshape(X, newshape=X.shape[0]))
    
    if afkmc2:
        # di = numpy.min(euclidean_distances(X, centers[0:1, :], squared=True), axis=1)*weights
    
        print(X)
        print(cntrs[0])
        
        di = numpy.fromiter((emd_cp(1, xi, cntrs[0]) for xi in X), numpy.float32, count=X.shape[0])
        
        q = di/numpy.sum(di) + weights/numpy.sum(weights)  # Only the potentials
        
        print(q)
        print(di)
        
    else:
        q = numpy.copy(weights)
    # Renormalize the proposal distribution
    q = q / numpy.sum(q)

    for i in range(k-1):
        print(i)
        t1 = time.time()
        # Draw the candidate indices
        cand_ind = random_state.choice(X.shape[0], size=(chain_length), p=q).astype(dtype=numpy.intp)
        
        # Extract the proposal probabilities
        q_cand = q[cand_ind].astype(numpy.double)
        #print(X[cand_ind].flatten())
        #x_view[:] = numpy.empty(cand_ind.shape[0], dtype=type(defff))
        #print(cand_ind)
        #print('asdfasdfasdfasdfasdfasdf')
        #print('asdfasdfljasdfl;kjasl;kfd')
        #print(X[cand_ind[0]])
        #x_view[:] = [X[X] for X in cand_ind]

        # Compute pairwise distances
        dist = [[emd_cp(1, xi, X[cand_ind[yi]]) for yi in range(len(cand_ind))] for xi in cntrs[0:(i+1)]]
        
        # Compute potentials
        # p_cand = numpy.min(dist, axis=1)*weights[cand_ind]
        p_cand = numpy.min(dist).astype(numpy.double)*weights[cand_ind].astype(numpy.double)
        # p_cand = dist[0]*weights[cand_ind]
        
        # Compute acceptance probabilities
        rand_a = random_state.random_sample(size=(chain_length))
        with cython.boundscheck(False), cython.wraparound(False), cython.cdivision(True):
            # Markov chain
            for j in range(q_cand.shape[0]):
                cand_prob = (p_cand[j])/(q_cand[j])
                if j == 0 or curr_prob == 0.0 or cand_prob/curr_prob > rand_a[j]:
                    # Init new chain             Metropolis-Hastings step
                    curr_ind = j
                    curr_prob = cand_prob
        # centers[i+1, :] = rel_row.todense().flatten() if sparse else rel_row
        cntrs[i+1] = X[cand_ind[curr_ind]]
        t2=time.time()
        t = t2-t1
        print('~' + str(t*(k - 1 - i)//60) + ' minutes until finished. ' + str(100*(i/(k-1)))[:4] + '% done     ', end = '\r')
    return cntrs



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
# Taken from stack overflow. Fast list combination generator for C(n, k)
#

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
    


# the extra arrays after v_values are empty arrays to be filled by the funciton.
# for best performance, it is up to you to define these empty arrays outside of heavy looping.
cdef float32 emd_c(int p, float32[:] u_values, float32[:] v_values, float32[:] all_values,float32[:] all_valuesa, float32[:] all_valuesaa, float32[:] deltas, u_weights=None, v_weights=None):
    r"""
    Compute, between two one-dimensional distributions :math:`u` and
    :math:`v`, whose respective CDFs are :math:`U` and :math:`V`, the
    statistical distance that is defined as:

    .. math::

        l_p(u, v) = \left( \int_{-\infty}^{+\infty} |U-V|^p \right)^{1/p}

    p is a positive parameter; p = 1 gives the Wasserstein distance, p = 2
    gives the energy distance.

    Parameters
    ----------
    u_values, v_values : array_like
        Values observed in the (empirical) distribution.
    u_weights, v_weights : array_like, optional
        Weight for each value. If unspecified, each value is assigned the same
        weight.
        `u_weights` (resp. `v_weights`) must have the same length as
        `u_values` (resp. `v_values`). If the weight sum differs from 1, it
        must still be positive and finite so that the weights can be normalized
        to sum to 1.

    Returns
    -------
    distance : float
        The computed distance between the distributions.

    Notes
    -----
    The input distributions can be empirical, therefore coming from samples
    whose values are effectively inputs of the function, or they can be seen as
    generalized functions, in which case they are weighted sums of Dirac delta
    functions located at the specified values.

    References
    ----------
    .. [1] Bellemare, Danihelka, Dabney, Mohamed, Lakshminarayanan, Hoyer,
           Munos "The Cramer Distance as a Solution to Biased Wasserstein
           Gradients" (2017). :arXiv:`1705.10743`.

    """
    #u_values, u_weights = _validate_distribution(u_values, u_weights)
    #v_values, v_weights = _validate_distribution(v_values, v_weights)

    insertion_sort_inplace_cython_float32(u_values)
    insertion_sort_inplace_cython_float32(v_values)

    # u_sorter = numpy.argsort(u_values)
    # v_sorter = numpy.argsort(v_values)

    cdef int i
    cdef int all_len = len(u_values) + len(v_values)
    
    #cdef float32[:] all_values = numpy.empty(all_len, dtype = numpy.float32)
    #cdef float32[:] all_valuesa = numpy.empty(all_len-1, dtype = numpy.float32)
    #cdef float32[:] all_valuesaa = numpy.empty(all_len-1, dtype = numpy.float32)
    #cdef float32[:] deltas = numpy.empty(all_len-1, dtype = numpy.float32)

    # all_values[:len(u_values)] = u_values
    # all_values[len(u_values):] = v_values

    insertion_sort_inplace_cython_float32(all_values)
    
    for i in range(len(u_values)):
        all_values[i] = u_values[i]
    
    for i in range(len(u_values), all_len):
        all_values[i] = v_values[i-len(u_values)]

    for i in range(all_len-1):
        deltas[i] = all_values[i+1] - all_values[i]
        



    return all_ind(u_values, v_values, all_values, all_valuesa, all_valuesaa, deltas)

cpdef float32 emd_cp(int p, float32[:] u_values, float32[:] v_values, u_weights=None, v_weights=None):
    r"""
    Compute, between two one-dimensional distributions :math:`u` and
    :math:`v`, whose respective CDFs are :math:`U` and :math:`V`, the
    statistical distance that is defined as:

    .. math::

        l_p(u, v) = \left( \int_{-\infty}^{+\infty} |U-V|^p \right)^{1/p}

    p is a positive parameter; p = 1 gives the Wasserstein distance, p = 2
    gives the energy distance.

    Parameters
    ----------
    u_values, v_values : array_like
        Values observed in the (empirical) distribution.
    u_weights, v_weights : array_like, optional
        Weight for each value. If unspecified, each value is assigned the same
        weight.
        `u_weights` (resp. `v_weights`) must have the same length as
        `u_values` (resp. `v_values`). If the weight sum differs from 1, it
        must still be positive and finite so that the weights can be normalized
        to sum to 1.

    Returns
    -------
    distance : float
        The computed distance between the distributions.

    Notes
    -----
    The input distributions can be empirical, therefore coming from samples
    whose values are effectively inputs of the function, or they can be seen as
    generalized functions, in which case they are weighted sums of Dirac delta
    functions located at the specified values.

    References
    ----------
    .. [1] Bellemare, Danihelka, Dabney, Mohamed, Lakshminarayanan, Hoyer,
           Munos "The Cramer Distance as a Solution to Biased Wasserstein
           Gradients" (2017). :arXiv:`1705.10743`.

    """
    #u_values, u_weights = _validate_distribution(u_values, u_weights)
    #v_values, v_weights = _validate_distribution(v_values, v_weights)

    insertion_sort_inplace_cython_float32(u_values)
    insertion_sort_inplace_cython_float32(v_values)

    # u_sorter = numpy.argsort(u_values)
    # v_sorter = numpy.argsort(v_values)

    
    all_len = len(u_values) + len(v_values)
    cdef float32[:] all_values = numpy.empty(all_len, dtype = numpy.float32)
    cdef float32[:] all_valuesa = numpy.empty(all_len-1, dtype = numpy.float32)
    cdef float32[:] all_valuesaa = numpy.empty(all_len-1, dtype = numpy.float32)
    cdef float32[:] deltas = numpy.empty(all_len-1, dtype = numpy.float32)

    all_values[:len(u_values)] = u_values
    all_values[len(u_values):] = v_values

    insertion_sort_inplace_cython_float32(all_values)
    

    for i in range(all_len-1):
        deltas[i] = all_values[i+1] - all_values[i]



    return all_ind(u_values, v_values, all_values, all_valuesa, all_valuesaa, deltas)


def _validate_distribution(values, weights):
    """
    Validate the values and weights from a distribution input of `cdf_distance`
    and return them as ndarray objects.

    Parameters
    ----------
    values : array_like
        Values observed in the (empirical) distribution.
    weights : array_like
        Weight for each value.

    Returns
    -------
    values : ndarray
        Values as ndarray.
    weights : ndarray
        Weights as ndarray.

    """
    # Validate the value array.
    values = numpy.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("Distribution can't be empty.")

    # Validate the weight array, if specified.
    if weights is not None:
        weights = numpy.asarray(weights, dtype=float)
        if len(weights) != len(values):
            raise ValueError('Value and weight array-likes for the same '
                             'empirical distribution must be of the same size.')
        if numpy.any(weights < 0):
            raise ValueError('All weights must be non-negative.')
        if not 0 < numpy.sum(weights) < numpy.inf:
            raise ValueError('Weight array-like sum must be positive and '
                             'finite. Set as None for an equal distribution of '
                             'weight.')

        return values, weights

    return values, None

cpdef emd(int p, u_values, v_values, u_weights=None, v_weights=None):
    r"""
    Compute, between two one-dimensional distributions :math:`u` and
    :math:`v`, whose respective CDFs are :math:`U` and :math:`V`, the
    statistical distance that is defined as:

    .. math::

        l_p(u, v) = \left( \int_{-\infty}^{+\infty} |U-V|^p \right)^{1/p}

    p is a positive parameter; p = 1 gives the Wasserstein distance, p = 2
    gives the energy distance.

    Parameters
    ----------
    u_values, v_values : array_like
        Values observed in the (empirical) distribution.
    u_weights, v_weights : array_like, optional
        Weight for each value. If unspecified, each value is assigned the same
        weight.
        `u_weights` (resp. `v_weights`) must have the same length as
        `u_values` (resp. `v_values`). If the weight sum differs from 1, it
        must still be positive and finite so that the weights can be normalized
        to sum to 1.

    Returns
    -------
    distance : float
        The computed distance between the distributions.

    Notes
    -----
    The input distributions can be empirical, therefore coming from samples
    whose values are effectively inputs of the function, or they can be seen as
    generalized functions, in which case they are weighted sums of Dirac delta
    functions located at the specified values.

    References
    ----------
    .. [1] Bellemare, Danihelka, Dabney, Mohamed, Lakshminarayanan, Hoyer,
           Munos "The Cramer Distance as a Solution to Biased Wasserstein
           Gradients" (2017). :arXiv:`1705.10743`.

    """
    u_values, u_weights = _validate_distribution(u_values, u_weights)
    v_values, v_weights = _validate_distribution(v_values, v_weights)

    

    u_sorter = numpy.argsort(u_values)
    v_sorter = numpy.argsort(v_values)

    all_values = numpy.concatenate((u_values, v_values))
    all_values.sort(kind='mergesort')

    # Compute the differences between pairs of successive values of u and v.
    deltas = numpy.diff(all_values)

    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], 'right')
    v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], 'right')

    print(u_cdf_indices)
    print(v_cdf_indices)
    print(deltas)

    # Calculate the CDFs of u and v using their weights, if specified.
    if u_weights is None:
        u_cdf = u_cdf_indices / u_values.size
    else:
        u_sorted_cumweights = numpy.concatenate(([0],
                                              numpy.cumsum(u_weights[u_sorter])))
        u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

    if v_weights is None:
        v_cdf = v_cdf_indices / v_values.size
    else:
        v_sorted_cumweights = numpy.concatenate(([0],
                                              numpy.cumsum(v_weights[v_sorter])))
        v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]

    # Compute the value of the integral based on the CDFs.
    # If p = 1 or p = 2, we avoid using numpy.power, which introduces an overhead
    # of about 15%.
    return numpy.sum(numpy.multiply(numpy.abs(u_cdf - v_cdf), deltas))
    





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
    z_memmap = numpy.memmap('results/river.npy', mode = 'r+', dtype = numpy.int16, shape = (x_shape * (y_shape - dupes), 3), offset = os )
    z_f_memmap = numpy.memmap('results/river_f.npy', mode = 'r+', dtype = numpy.float32, shape = (x_shape * (y_shape - dupes), 1), offset = os//3 * 2)
    #mp_memmap = numpy.memmap('results/map.npy', mode = 'r+', dtype = numpy.ulonglong, shape = (x_shape * (y_shape - dupes), 1), offset = os//3 * 2)
    
    cdef int16 [:, :] z_view = z_memmap
    cdef numpy.float32_t [:, :] z_f_view = z_f_memmap
    
    #cdef unsigned long long [:, :] mp_view = mp_memmap

    cdef numpy.ndarray[int16, ndim=1] oh = numpy.empty(7, dtype=numpy.int16)
    cdef int16[:] oh_view = oh[:]

    cdef unsigned long long one = 1
    cdef unsigned long long key
    for i in range(x_shape):
        t1=time.time()
        for j in range(y.shape[0]):
            oh_view[:2] = x[i]
            oh_view[2:] = y[j]
            if(not contains_duplicates(oh_view)):
                
                get_ehs_fast(oh_view, z_view[cd])

                key = (one << oh_view[0]) | (one << oh_view[1]) | (one << oh_view[2]) | (one << oh_view[3]) | (one << oh_view[4]) | (one << oh_view[5]) | (one << oh_view[6]) 
                
                
                z_f_view[cd] = (z_view[cd][1]+.5*z_view[cd][0]) / (z_view[cd][1]+z_view[cd][0]+z_view[cd][2])

                #mp_view[cd] = key

                cd += 1
            
        t2=time.time()
        t = t2-t1
        print('~' + str(t*(x_shape - i)//60) + ' minutes until finished. ' + str(100*(i/x_shape))[:4] + '% done     ', end = '\r')
    # mp_memmap.flush()
    z_f_memmap.flush()
    z_memmap.flush()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void insertion_sort_inplace_cython_float32(float32[:] A):
    cdef: 
        int i, j
        numpy.float32_t key
        int length_A = A.shape[0]

    for j in range(1, length_A):
        key = A[j]
        i = j - 1
        while (i >= 0) & (A[i] > key):
            A[i + 1] = A[i]
            i = i - 1
        A[i + 1] = key


#set elents in A to the index they appear in B
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef float32 all_ind(float32[:] A, float32[:] AA, float32[:] B, float32[:] BB, float32[:] BBB, float32[:] deltas):
    cdef: 
        int32 i, j
        numpy.float32_t key, keyy, pd
        int32 length_A = len(A)
        int32 length_AA = len(AA)
        int32 length_B = len(B)

    pd = 0

    for j in range(length_B - 1):
        key = B[j]
        i = 0
        while (i < length_A) & (A[i] <= key):
            i = i + 1
        BB[j] = i/length_A

        i = 0
        while (i < length_AA) & (AA[i] <= key):
            i = i + 1
        BBB[j] = i/length_AA

    for i in range(length_B-1):

        if(BB[i]>=BBB[i]):
            pd += (BB[i] - BBB[i]) * deltas[i]
        else:
            pd += (BBB[i] - BB[i]) * deltas[i]
    return pd
        


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
cdef prob_dist_fun(int16[:, :] x, int16[:, :] y, float32[:, :] dist, float32[:, :] cntrs, int[:] lbs, int64 dupes, mp, boolean turn, int64 os): #single byte offset. multiply by #cols and #bytes in data type

    cdef int T_CARDS
    cdef int N_CARDS
    if turn:
        T_CARDS = 5
        N_CARDS = 16
    else:
        T_CARDS = 4
        N_CARDS = 16
        
    
    #cdef int16[:, :] x = nump2(N_CARDS, 2)
    #cdef int16[:, :] y = nump2(N_CARDS, T_CARDS)
    print(T_CARDS)

    cdef numpy.ndarray[int16, ndim=1] oh = numpy.empty(T_CARDS+1, dtype=numpy.int16)
    cdef numpy.ndarray[int16, ndim=1] oh_z = numpy.empty(T_CARDS+2, dtype=numpy.int16)
    cdef numpy.ndarray[int16, ndim=1] oh_z_tmp = numpy.empty(T_CARDS + 2, dtype=numpy.int16)

    
    cdef numpy.ndarray[float32, ndim=2] prob_dist

    cdef int16[:] oh_view = oh[:]
    cdef int16[:] oh_z_view = oh_z[:]
    cdef int16[:] oh_z_view_tmp = oh_z_tmp[:]
    
    cdef int c, i, j, tt, k
    cdef long t

    cdef unsigned long long [:, :] mp_turn_view
    cdef numpy.ndarray[float32, ndim=1] mp_z

    yy = nump2(N_CARDS, T_CARDS)
    if(turn):
        ndupes = num_dupes(x, yy)
    else:
        ndupes = num_dupes_turn(x, yy)
     
    

    mp_z = numpy.zeros(len(dist[0]), dtype = numpy.float32)
    
    cdef float32[:] mp_z_view = mp_z[:] 
    

    print(dist)

    # for each possible remaining card, we will store the subsequent index of the river cluster to which it responds.
    if turn:
        prob_dist = numpy.memmap('results/prob_dist_TURN.npy', mode = 'r+', dtype = numpy.float32, shape = (x.shape[0] * (y.shape[0] - dupes), N_CARDS - T_CARDS - 1), offset = os * 4 * (N_CARDS - T_CARDS-1))[:] # 10 col
    else:
        prob_dist = numpy.memmap('results/prob_dist_FLOP.npy', mode = 'r+', dtype = numpy.float32, shape = (x.shape[0] * (y.shape[0] - dupes), N_CARDS - T_CARDS - 1), offset = os * 4 * (N_CARDS - T_CARDS-1))[:] # 11 col -- both will need to change according to proper histogram schematics..
    
    cdef float32[:, :] prob_dist_memview = prob_dist[:]
    cdef float32[:] prob_dist_sing_memview = numpy.empty(N_CARDS-T_CARDS-1, dtype = numpy.float32)[:]
    cd = 0

    cdef float32[:] all_values = numpy.empty(N_CARDS-T_CARDS-1 + len(dist[0]), dtype = numpy.float32)
    cdef float32[:] all_valuesa = numpy.empty(N_CARDS-T_CARDS-2 + len(dist[0]), dtype = numpy.float32)
    cdef float32[:] all_valuesaa = numpy.empty(N_CARDS-T_CARDS-2 + len(dist[0]), dtype = numpy.float32)
    cdef float32[:] deltaaa = numpy.empty(N_CARDS-T_CARDS-2 + len(dist[0]), dtype = numpy.float32)


    cdef unsigned long long one = 1
    cdef unsigned long long key

    for i in range(x.shape[0]):
        # load current portion of dataset to memory using the offset. 
        # fluff_view[:, :] = numpy.memmap('results/fluffy.npy', mode = 'c', dtype = numpy.float32, shape = (y_shape_river[0], 1), offset = i * y_shape_river[0] * 8)[:]
        t1=time.time()
        
        for j in range(y.shape[0]):
            oh_view[:2] = x[i][:]
            oh_view[2:] = y[j][:]
            if(not ((turn and contains_duplicates_turn(oh_view)) or (not turn and contains_duplicates_flop(oh_view)))):
                oh_z_view[:T_CARDS+1] = oh_view
                oh_z_view_tmp[:] = oh_z_view

                prob_dist_sing_memview[:] = prob_dist_memview[cd][:]
                
                c = 0
                # print(cd)
                for k in range(N_CARDS):
                    if(not ((turn and contains_turn(oh_view, k)) or (not turn and contains_flop(oh_view, k)))):
                        ## for each datapoint (float value for winning), find the best center given 
                        
                        
                        oh_z_view[T_CARDS + 1] = k

                        # print()
                        # print()

                        # must sort table cards before using the mapping.. 
                        #insertion_sort_inplace_cython_int16(oh_z_view[2:])
                        
                        # print([oh_z_view[xxx] for xxx in range(T_CARDS+2)])
                        # print([oh_z_view_tmp[xxx] for xxx in range(T_CARDS+2)])

                        if(turn):
                            key = (one << oh_z_view[0])  | (one << oh_z_view[1]) | (one << oh_z_view[2])  | (one << oh_z_view[3])  | (one << oh_z_view[4])  | (one << oh_z_view[5])  | (one << oh_z_view[6])
                        else:
                            key = (one << oh_z_view[0])  | (one << oh_z_view[1]) | (one << oh_z_view[2])  | (one << oh_z_view[3])  | (one << oh_z_view[4]) | (one << oh_z_view[5])
                        
                        # offset key returned by os. 

                        # prob_dist_memview[cd][c] = emd_c(1, dist[mp[key]], cntrs[lbs[mp[key]]])

                        
                        #reward distance from losing.. temp logic until i understand paper
                        prob_dist_sing_memview[c] = emd_c(1, mp_z_view, cntrs[lbs[mp[key]]], all_values, all_valuesa, all_valuesaa, deltaaa)

                        #print([dist[mp[key]][jjj] for jjj in range(len(dist[mp[key]]))])
                        
                        #print([cntrs[lbs[mp[key]]][jjj] for jjj in range(len(cntrs[lbs[mp[key]]]))])
                        
                        

                        #oh_z_view[:] = oh_z_view_tmp
                        c += 1
                
                insertion_sort_inplace_cython_float32(prob_dist_sing_memview)
                prob_dist_memview[cd] = prob_dist_sing_memview[:]
                #print([prob_dist_memview[cd][jjj] for jjj in range(len(prob_dist_memview[cd]))])
                cd += 1
        
        t2=time.time()
        t = t2-t1
        print('~' + str(t*(x.shape[0] - i)//60) + ' minutes until finished. ' + str(100*(i/x.shape[0]))[:4] + '% done     ', end = '\r')


    return dupes


@cython.boundscheck(False) 
@cython.wraparound(False)
cpdef flop_ehs(n, k, threads, dupes, new_file=False):
    adjcntrs = numpy.load('results/adjcntrs_TURN.npy', mmap_mode = 'c')
    lbls = numpy.load('results/lbls_TURN.npy', mmap_mode = 'c')
    
    cdef numpy.ndarray[float32, ndim = 2] cntrs = adjcntrs
    cdef int[:] lbs = lbls

    mp = {}
    cdef numpy.ndarray[int16, ndim=1] oh = numpy.empty(k+1, dtype=numpy.int16)
    cdef int16[:] oh_view = oh[:]


    cdef int16[:, :] x = nump2(n, 2)
    cdef int16[:, :] y = nump2(n, k-1)

    

    dupes = num_dupes_turn(x, y)

    
    if(new_file):
        flop_dist = numpy.memmap('results/prob_dist_FLOP.npy', mode = 'w+', dtype = numpy.float32, shape = (x.shape[0] * (y.shape[0] - dupes), n - k-2))
        flop_dist.flush()


    cdef unsigned long long one = 1
    cdef int cd = 0
    
    for i in range(x.shape[0]):
        # load current portion of dataset to memory using the offset. 
        # fluff_view[:, :] = numpy.memmap('results/fluffy.npy', mode = 'c', dtype = numpy.float32, shape = (y_shape_river[0], 1), offset = i * y_shape_river[0] * 8)[:]
        t1=time.time()
        for j in range(y.shape[0]):
            oh_view[:2] = x[i]
            oh_view[2:] = y[j]
            if(not contains_duplicates_turn(oh_view)):
                mp[(one << oh_view[0]) | (one << oh_view[1]) | (one <<oh_view[2]) | (one <<oh_view[3]) | (one <<oh_view[4]) | (one <<oh_view[5])] = cd
                
                cd+=1





    cdef numpy.ndarray[float32, ndim=2] dist = numpy.memmap('results/prob_dist_TURN.npy', mode = 'c', dtype = numpy.float32, shape = (x.shape[0] * (y.shape[0] - dupes), n - k - 1))
    cdef float32[:, :] dist_view = dist[:]

    
    y = nump2(n, k-2)
    dupes = num_dupes_flop(x, y)

    chunksize = len(x) // (threads-1)
    print('Starting FLOP EHS')    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # change threads to appropriate number of workers for your system
        futures = []
        for i in range(threads-1):
            strt = i * chunksize
            stp = ((i + 1) * chunksize) if i != (threads - 2) else len(x)
            print(strt, end = ' ')
            print(stp)
            prob_dist_fun(x[strt:stp], y, dist, cntrs, lbs, dupes, mp, False, strt * (y.shape[0]-dupes))
            #futures.append(executor.submit(prob_dist_fun, x[strt:stp], y, adjcntrs, lbls, dupes, mp, False, strt * (y.shape[0]-dupes)))
        #concurrent.futures.wait(futures)

        #output = [f.result() for f in futures]
    
    return dupes

@cython.boundscheck(False) 
@cython.wraparound(False)
cpdef turn_ehs(n, k, threads, dupes, new_file=False):
    adjcntrs = numpy.load('results/adjcntrs.npy', mmap_mode = 'c')
    lbls = numpy.load('results/lbls.npy', mmap_mode = 'c')
    #print(adjcntrs)
    #print(lbls)
    #print(min(lbls))
    #print(max(lbls))
    
    cdef float32[:, :] cntrs = adjcntrs
    cdef int[:] lbs = lbls

    cdef int16[:, :] x = nump2(n, 2)
    cdef int16[:, :] y = nump2(n, k)

    dupes = num_dupes(x, y)
    
    cdef unsigned long long one = 1
    cdef unsigned long long keyy
    cdef int cd = 0
    mp = {}
    cdef numpy.ndarray[int16, ndim=1] oh = numpy.empty(k+2, dtype=numpy.int16)
    cdef int16[:] oh_view = oh[:]
    
    for i in range(x.shape[0]):
        # load current portion of dataset to memory using the offset. 
        # fluff_view[:, :] = numpy.memmap('results/fluffy.npy', mode = 'c', dtype = numpy.float32, shape = (y_shape_river[0], 1), offset = i * y_shape_river[0] * 8)[:]
        t1=time.time()
        for j in range(y.shape[0]):
            oh_view[:2] = x[i]
            oh_view[2:] = y[j]
            if(not contains_duplicates(oh_view)):
                
                keyy = (one << oh_view[0]) | (one << oh_view[1]) | (one <<oh_view[2]) | (one <<oh_view[3]) | (one <<oh_view[4]) | (one <<oh_view[5]) | (one <<oh_view[6])
                mp[keyy] = cd
                cd+=1


    cdef numpy.ndarray[float32, ndim=2] dist = numpy.memmap('results/river_f.npy', mode = 'c', dtype = numpy.float32, shape = (x.shape[0] * (y.shape[0] - dupes), 1))

    cdef float32[:, :] dist_view = dist[:]
    #numpy.ndarray[float32, ndim=2]
    # begin turn considerations
    
    x = nump2(n, 2)
    y = nump2(n, k-1)

    dupes = num_dupes_turn(x, y)
    print(dupes)
    if(new_file):
        prob_dist = numpy.memmap('results/prob_dist_TURN.npy', mode = 'w+', dtype = numpy.float32, shape = (x.shape[0] * (y.shape[0] - dupes), n - k - 1))
        
        prob_dist.flush()


    

    chunksize = len(x) // (threads-1)
    print('Starting TURN EHS')    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # change threads to appropriate number of workers for your system
        futures = []
        for i in range(threads-1):
            strt = i * chunksize
            stp = ((i + 1) * chunksize) if i != (threads - 2) else len(x)
            print(strt, end = ' ')
            print(stp)
            prob_dist_fun(x[strt:stp], y, dist[:], cntrs, lbs, dupes, mp, True, strt * (y.shape[0]-dupes))
            #futures.append(executor.submit(prob_dist_fun, x[strt:stp], y, adjcntrs, lbls, dupes, mp, True, strt * (y.shape[0]-dupes)))
        #concurrent.futures.wait(futures)

        output = [f.result() for f in futures]
    
    return dupes



@cython.boundscheck(False) 
@cython.wraparound(False)
cdef num_dupes_flop(int16[:, :] x, int16[:, :] y):
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
        oh[:2] = x[0][:]
        oh[2:] = y[j][:]
        if(contains_duplicates_flop(oh)):
            cd += 1
    return cd

@cython.boundscheck(False) 
@cython.wraparound(False)
cdef num_dupes_turn(int16[:, :] x, int16[:, :] y):
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
        oh[:2] = x[0]
        oh[2:] = y[j]
        if(contains_duplicates_turn(oh)):
            cd += 1
    return cd

# @cython.profile(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef num_dupes(int16[:, :] x, int16[:, :] y):
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
        oh[:2] = x[0]
        oh[2:] = y[j]
        if(contains_duplicates(oh)):
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
        #mp = numpy.memmap('results/map.npy', mode = 'w+', dtype = numpy.ulonglong, shape = ((y.shape[0] - dupes) * x.shape[0], 1))
            
        z.flush()
        z_f.flush()
        #mp.flush()

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

    