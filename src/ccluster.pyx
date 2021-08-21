import time
import numpy
import random



cimport numpy
cimport cython
from libc.stdio cimport FILE, fopen, fwrite, fscanf, fclose, fprintf

ctypedef numpy.uint8_t uint8
ctypedef numpy.int16_t int16
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


@cython.boundscheck(False) 
@cython.wraparound(False)
cdef unsigned long long cards_to_mask(int16[:] py_cards) nogil:
    cdef numpy.ulonglong_t cards = 0
    for p in range(7):
        cards |= (<unsigned long long>1) << (py_cards[p])
    return cards


#################################################################################################
# My code below.
#################################################################################################

# 
# See if a card appears twice in the hand/table combo. Can be faster.
# 
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
# Check if the hand/board combo (XX) contains a given card (comp)
# 
@cython.boundscheck(False)
@cython.wraparound(False)
cdef numpy.npy_bool contains(int16[:] XX, int comp) nogil:
    cdef unsigned int count
    cdef unsigned int length = 7
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
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef int16[:] get_ehs_fast(int16[:] j):
    cdef int16 twl_tiewinloss[3]
    cdef int16 x[45]
    cdef int16 i, k
    cdef int hero, v, c

    cdef unsigned int seven = 7
    
    twl_tiewinloss[0] = 0
    twl_tiewinloss[1] = 0
    twl_tiewinloss[2] = 0
    
    c = 0
    for i in range(52):
        if not contains(j, i):
            x[c] = i
            c += 1

    hero = cy_evaluate(cards_to_mask(j), seven)

    for i in range(0, 45-1):
        for k in range(i+1, 45):
            j[0] = x[i]
            j[1] = x[k]
            v = cy_evaluate(cards_to_mask(j), seven)
            if(hero > v):
                twl_tiewinloss[1] += 1
            elif(v > hero):
                twl_tiewinloss[2] += 1
            else:
                twl_tiewinloss[0] += 1
    
    return twl_tiewinloss


# 
# All major computation is done in C. Only remaining overhead is in the 'get_ehs_fast'
# function call. Solution is likely to send the whole z_view row and let get_ehs_fast
# return the same row.
# 
@cython.boundscheck(False) 
@cython.wraparound(False)
cpdef do_calc(int16[:, :] x, int16[:, :] y, int id):
    cdef unsigned long long int total = 0
    cdef numpy.int64_t i, j, c
    cdef double t1, t2, t
    cdef numpy.int64_t x_shape = x.shape[0]
    cdef numpy.int64_t y_shape = y.shape[0]
    
    
    c = 0
    
    cdef numpy.ndarray[int16, ndim=2] z = numpy.empty((x_shape * y_shape, x.shape[1] + y.shape[1] + 3), dtype=numpy.int16)
    cdef int16 [:, :] z_view = z
    cdef numpy.ndarray[int16, ndim=1] oh = numpy.empty(7, dtype=numpy.int16)
    cdef int16 [:] oh_view = oh


    for i in range(x_shape):
        t1=time.time()
        for j in range(y_shape):
            oh_view[:2] = x[i]
            oh_view[2:] = y[j]
            if(not contains_duplicates(oh_view)):
                z_view[c][:7] =  oh_view
                z_view[c][7:] = get_ehs_fast(oh_view)
            c += 1
        t2=time.time()
        t = t2-t1
        print('~' + str(t*(x_shape - i)//60) + ' minutes until finished. ' + str(100*(i/x_shape))[:4] + '% done     ', end = '\r')
    numpy.save('results/zfin-'+str(id), numpy.asarray(z_view))

    