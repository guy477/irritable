from typing import Dict, Tuple, List
import operator
import math

from poker_ai.poker.card import Card


def make_starting_hand_lossless(starting_hand, short_deck) -> int:
    ranks = []
    suits = []
    for card in starting_hand:
        ranks.append(card.rank_int)
        suits.append(card.suit)
    if len(set(suits)) == 1:
        suited = True
    else:
        suited = False
    if all(c_rank == 14 for c_rank in ranks):
        return 0
    elif all(c_rank == 13 for c_rank in ranks):
        return 1
    elif all(c_rank == 12 for c_rank in ranks):
        return 2
    elif all(c_rank == 11 for c_rank in ranks):
        return 3
    elif all(c_rank == 10 for c_rank in ranks):
        return 4
    elif 14 in ranks and 13 in ranks:
        return 5 if suited else 15
    elif 14 in ranks and 12 in ranks:
        return 6 if suited else 16
    elif 14 in ranks and 11 in ranks:
        return 7 if suited else 17
    elif 14 in ranks and 10 in ranks:
        return 8 if suited else 18
    elif 13 in ranks and 12 in ranks:
        return 9 if suited else 19
    elif 13 in ranks and 11 in ranks:
        return 10 if suited else 20
    elif 13 in ranks and 10 in ranks:
        return 11 if suited else 21
    elif 12 in ranks and 11 in ranks:
        return 12 if suited else 22
    elif 12 in ranks and 10 in ranks:
        return 13 if suited else 23
    elif 11 in ranks and 10 in ranks:
        return 14 if suited else 24


def make_starting_hand_lossless_full(starting_hand, full_deck) -> int:
    
    ranks = []
    suits = []
    c = ''
    for card in starting_hand:
        ranks.append(card.rank)
        if(ranks[-1] == '10'):
            ranks[-1] = 'T'
        elif(ranks[-1].__len__() > 1):
            ranks[-1] = ranks[-1][0].upper()
        c = c + ranks[-1]
        suits.append(card.suit)

    if len(set(suits)) == 1:
        c = c + 's'
    elif len(set(ranks)) == 2:
        c = c + 'o'
    # print(c)
    order = 'AA,KK,QQ,AKs,JJ,AQs,KQs,AJs,KJs,TT,AKo,ATs,QJs,KTs,QTs,JTs,99,AQo,A9s,KQo,88,K9s,T9s,A8s,Q9s,J9s,AJo,A5s,77,A7s,KJo,A4s,A3s,A6s,QJo,66,K8s,T8s,A2s,98s,J8s,ATo,Q8s,K7s,KTo,55,JTo,87s,QTo,44,33,22,K6s,97s,K5s,76s,T7s,K4s,K3s,K2s,Q7s,86s,65s,J7s,54s,Q6s,75s,96s,Q5s,64s,Q4s,Q3s,T9o,T6s,Q2s,A9o,53s,85s,J6s,J9o,K9o,J5s,Q9o,43s,74s,J4s,J3s,95s,J2s,63s,A8o,52s,T5s,84s,T4s,T3s,42s,T2s,98o,T8o,A5o,A7o,73s,A4o,32s,94s,93s,J8o,A3o,62s,92s,K8o,A6o,87o,Q8o,83s,A2o,82s,97o,72s,76o,K7o,65o,T7o,K6o,86o,54o,K5o,J7o,75o,Q7o,K4o,K3o,96o,K2o,64o,Q6o,53o,85o,T6o,Q5o,43o,Q4o,Q3o,74o,Q2o,J6o,63o,J5o,95o,52o,J4o,J3o,42o,J2o,84o,T5o,T4o,32o,T3o,73o,T2o,62o,94o,93o,92o,83o,82o,72o'.split(',')

    return order.index(c)

def compute_preflop_lossless_abstraction(builder) -> Dict[Tuple[Card, Card], int]:
    """Compute the preflop abstraction dictionary.

    Only works for the short deck presently.
    """
    # Making sure this is 20 card deck with 2-9 removed
    # allowed_ranks = {10, 11, 12, 13, 14}
    # found_ranks = set([c.rank_int for c in builder._cards])
    # if found_ranks != allowed_ranks:
    #     raise ValueError(
    #         f"Preflop lossless abstraction only works for a short deck with "
    #         f"ranks [10, jack, queen, king, ace]. What was specified="
    #         f"{found_ranks} doesn't equal what is allowed={allowed_ranks}"
    #     )
    # Getting combos and indexing with lossless abstraction

    # Expand to allow lossless abstraction for full 52 card deck
    preflop_lossless: Dict[Tuple[Card, Card], int] = {}
    for starting_hand in builder.starting_hands:
        starting_hand = sorted(
            list(starting_hand),
            key=operator.attrgetter("eval_card"),
            reverse=True
        )
        preflop_lossless[tuple(starting_hand)] = make_starting_hand_lossless_full(
            starting_hand, builder
        )
    return preflop_lossless
