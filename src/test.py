import numpy as np
from eval7 import Card, evaluate
# from pokereval.card import Card
# from pokereval.hand_evaluator import HandEvaluator
import random
import holdem_calc
import time

def get_card(x):
    rank = {0 : '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8', 7: '9', 8: 'T', 9: 'J', 10: 'Q', 11: 'K', 12: 'A'}
    suit = {0: 's', 1: 'c', 2: 'h', 3: 'd'}
    
    return (rank[x%13] + suit[x//13])


# def get_ehs_fast_pokereval(board, hand):
#     x = list(range(52))
#     j = board + hand
#     for i in j:
#         x.remove(i)
#     print(board)
#     board = [Card(x%13 + 2, 1+x//13) for x in board]
#     hero = HandEvaluator.evaluate_hand(hand = [Card(x%13 + 2, 1+x//13) for x in hand], board = board)
    
#     vil = -1
#     wlt = [0, 0, 0]
#     for i in range(0, len(x)-1):
#         for k in range(i+1, len(x)):
            
#             vil = HandEvaluator.evaluate_hand(hand = [Card(s%13 + 2, 1+s//13) for s in [x[i], x[k]]], board = board)
#             if(hero > vil):
#                 wlt[1] += 1
#             elif(vil > hero):
#                 wlt[-1] += 1
#             else:
#                 wlt[0] += 1
    
#     s = sum(wlt)
#     wlt[0] /= s
#     wlt[1] /= s
#     wlt[2] /= s
#     return wlt


def get_ehs_fast(combo):
    x = list(range(52))
    j = combo
    for i in j:
        x.remove(i)

    hero = evaluate([Card(get_card(s)) for s in j])
    vil = -1
    wlt = [0, 0, 0]
    for i in range(0, len(x)-1):
        for k in range(i+1, len(x)):
            
            vil = evaluate([Card(get_card(s)) for s in j[2:] + [x[i], x[k]]])
            if(hero > vil):
                wlt[1] += 1
            elif(vil > hero):
                wlt[-1] += 1
            else:
                wlt[0] += 1
    
    s = sum(wlt)
    wlt[0] /= s
    wlt[1] /= s
    wlt[2] /= s
    return wlt

def get_ehs(board, hand):
    x = list(range(52))
    j = board + hand

    for i in j:
        x.remove(i)
    wlt = [0, 0, 0]
    for i in range(0, len(x)-1):
        for j in range(i+1, len(x)):
            # print([get_card(x) for x in board])
            # print([get_card(x) for x in hand + [x[i], x[j]]])
            w = holdem_calc.calculate(board = [get_card(k) for k in board], exact = True, hole_cards = [get_card(x) for x in hand + [x[i], x[j]]], verbose = False, num = 10, input_file = None)
            wlt[0] += w[0]
            wlt[1] += w[1]
            wlt[2] += w[2]
            
    s = sum(wlt)
    wlt[0] /= s
    wlt[1] /= s
    wlt[2] /= s
    return wlt

program_starts = time.time()
# print(float(b'00111110101101101101101101101110'))
for i in range(100):
    combo = [43, 12, 5, 34, 23] + [8, 9]
    x = list(range(52))
    j = combo
    for i in j:
        x.remove(i)

    hero = evaluate([Card(get_card(s)) for s in j])
    vil = -1
    wlt = [0, 0, 0]
    for i in range(0, len(x)-1):
        for k in range(i+1, len(x)):
            
            vil = evaluate([Card(get_card(s)) for s in j[2:] + [x[i], x[k]]])
            if(hero > vil):
                wlt[1] += 1
            elif(vil > hero):
                wlt[-1] += 1
            else:
                wlt[0] += 1
    # get_ehs_fast(combo = [43, 12, 5, 34, 23] + [8, 9])
    # print(get_ehs_fast_pokereval(board = [43, 12, 5, 34, 23], hand = [8, 9]))
        
now = time.time()
print("It has been {0} seconds since the loop started".format(now - program_starts))



# # # def nump2(n, k):
# #     # a = np.ones((k, n-k+1), dtype=np.uint8)
# #     # a[0] = np.arange(n-k+1)
# #     # for j in range(1, k):
# #         # reps = (n-k+j) - a[j-1]
# #         # a = np.repeat(a, reps, axis=1)
# #         # ind = np.add.accumulate(reps)
# #         # a[j, ind[:-1]] = 1-reps[1:]
# #         # a[j, 0] = j
# #         # a[j] = np.add.accumulate(a[j])
# #     # return a.T


# # # def get_card(x):
# #     # rank = {0 : '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8', 7: '9', 8: 'T', 9: 'J', 10: 'Q', 11: 'K', 12: 'A'}
# #     # suit = {0: 's', 1: 'c', 2: 'h', 3: 'd'}
# #     # return (rank[x%13] + suit[x//13])


# # # def map_indices(arr):
    
# #     # mp = {}    
# #     # c = 0

# #     # if(len(arr[0]) == 5):
# #         # for x in arr.tolist():
# #             # mp[str(x[0]) + str(x[1]) + str(x[2]) + str(x[3]) + str(x[4])] = c
# #             # c += 1
# #     # elif(len(arr[0]) == 4):

# #         # for x in arr.tolist():
# #             # mp[str(x[0]) + str(x[1]) + str(x[2]) + str(x[3])] = c
# #             # c += 1

# #     # else:

# #         # for x in arr.tolist():
# #             # mp[str(x[0]) + str(x[1]) + str(x[2])] = c
# #             # c += 1
# #     # return mp


# # # def map_indices_by_hand(arr):
    
# #     # mp = {}    
# #     # c = 0
# #     # for x in arr.tolist():
# #         # mp[str(x[0]) + str(x[1])] = c
# #         # c += 1
# #     # return mp



# # # x = nump2(52, 2)

# # # y = nump2(52, 5)
# # # yy = nump2(52, 4)
# # # yyy = nump2(52, 3)

# # # mp_hand = map_indices_by_hand(x)
# # # mp_table_river = map_indices(y)
# # # mp_table_turn = map_indices(yy)
# # # mp_table_flop = map_indices(yyy)




# # print([get_card(x) for x in [0, 1, 2, 3, 4]])
# # print([get_card(x) for x in [5, 6]] )
# # print(holdem_calc.calculate(board = [get_card(x) for x in [43, 12, 5, 34, 23]], exact = True, hole_cards = [get_card(x) for x in [9, 8]], verbose = True, num = 10, input_file = None))