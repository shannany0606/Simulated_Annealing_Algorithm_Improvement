import random
import math
from collections import deque
from common import get_cost
from common import *

def SA6(n, points, adj_mat, max_tnm, term_count_1, term_count_2, t_0, alpha):
    """
    :param n: number of vertices
    :param adj_mat: adjacency matrix
    :param tb_size: >=0, max length of tb_list
    :param max_tnm: how many candidates picked in tournament selection
    :param mut_md: [get_sol, get delta], method of mutation, e.g. swap, 2-opt
    :param term_count_1: inner loop termination flag
    :param term_count_2: outer loop termination flag
    :param t_0: starting temperature
    :param alpha: cooling parameter
    """
    # initialization
    # sol = list(range(n))
    # random.shuffle(sol)  # e.g. [0,1,...,n]

    sol = Init(points, n, 1, adj_mat).newCreate()
    cost = get_cost(n, adj_mat, sol)
    
    best_sol = sol.copy()
    best_cost = cost
    
    t = t_0
    
    data = {'cost': deque([]), 'best_cost': deque([]),
            'sol': deque([]), 'best_sol': deque([])}
    count_2 = 0  # outer loop count
    while True:
        count_1 = 0  # inner loop count
        best_inner_sol = sol
        best_inner_cost = cost
        while True:
            last_sol = sol
            last_cost = cost

            Select = random.randint(1,4)
            if Select == 1 or Select == 2: mut_md = [get_new_sol_2opt, get_delta_2opt]
            # if Select == 2: mut_md = [get_new_sol_swap, get_delta_swap]
            if Select == 3 or Select == 4: mut_md = [get_new_sol_move, get_delta_move]

            sol, cost = tnm_selection(n, adj_mat, sol, max_tnm, mut_md)
            # mention the iteratively variable 'sol'!
            if cost > last_cost and math.exp((last_cost - cost) / t) < random.random():
                sol = last_sol
                cost = last_cost
            if cost < best_inner_cost:
                best_inner_sol = sol
                best_inner_cost = cost
                count_1 = 0
            else:
                count_1 += 1
            data['cost'].append(cost)
            data['best_cost'].append(best_cost)
            data['sol'].append(sol)
            data['best_sol'].append(best_sol)
            if count_1 > term_count_1:
                break
        # end of inner loop
        # get best_inner_sol < sol
        t = alpha * t
        if best_inner_cost < best_cost:
            best_sol = best_inner_sol
            best_cost = best_inner_cost
            count_2 = 0
        else:
            count_2 += 1
        if count_2 > term_count_2:
            break
    return best_sol, best_cost, data
