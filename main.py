import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
import numpy as np
from pprint import pprint
import time
import math
from tqdm import tqdm

from common import *
from GA import GA

import SA1
import SA2
import SA3
import SA4
import SA5
import SA6
import SA6

import os
if not os.path.exists('results'):
    os.makedirs('results')

fileName = ["a280.tsp","d198.tsp","d493.tsp","eil101.tsp","gil262.tsp","gr431.tsp","kroA150.tsp","p654.tsp","u724.tsp","ch130.tsp","eil76.tsp"]
optDis = [2579 , 15780 , 35002 , 629 , 2378 , 171414 , 26524 , 34643 , 41910 , 6110 , 538]

print("1.a280")
print("2.d198")
print("3.d493")
print("4.eil101")
print("5.gil262")
print("6.gr431")
print("7.kroA150")
print("8.p654")
print("9.u724")
print("10.ch130")
print("11.eil76")
print("请输入您选择的测试集序号",end=":")
choose=int(input())
#choose=1
choose-=1
while choose>=len(fileName):
    print("序号不在1~11以内，请重新输入")

# load data
with open('data/'+fileName[choose]) as fp: 
    lines=fp.readlines()
    pos = [[float(x) for x in s.split()[1:]] + [int(x) for x in s.split()[:1]] for s in lines]
    dim=len(lines)
    graph=np.zeros((dim,3))  # 0:city index 1:x 2:y
    distmat=np.zeros((dim,dim))
    for i in range(dim):
        for j,pox in enumerate(filter(lambda x: x and x.strip(),lines[i].split(' '))):
            graph[i][j]=float(pox)
    for i in range(dim):
        for j in range(i,dim):
            if i==j:
                distmat[i][j]=float('inf')
            else:
                distmat[i][j]=distmat[j][i]=np.linalg.norm(graph[i,1:]-graph[j,1:])

opt_cost = optDis[choose]  # get result from tsp_gurobi.py
num_tests = 100 # number of iid tests
result = {'best_sol': [], 'best_cost': math.inf, 'best_gap': math.inf,
          'cost': [0] * num_tests, 'time': [0] * num_tests,
          'avg_cost': math.inf, 'avg_gap': math.inf, 'cost_std': math.inf,
          'avg_time': math.inf, 'time_std': math.inf}
best_cost = math.inf
best_sol = []
data = {}

# set method

# method = 'GA'  # genetic algorithm
# method = 'SA'  # simulated annealing
# method = 'ISA-2-opt' #Use 2-opt operator
# method = 'ISA-LOOP' #Inner Loop
# method = 'ISA-MM' # Multiple Mutation
method = 'ISA-TP' # Tuning Parameters
# method = 'ISA-OIS' # Optimized Initial Solution

# run and visualization

method_name = ''
for _ in tqdm(range(num_tests)):
    start = time.time()
    if method == 'GA':
        method_name = 'Genetic Algorithm'
        best_sol, best_cost, data = GA(pos, distmat, 
                                        n_pop=200, 
                                        r_cross=0.8, 
                                        r_mut=0.3,
                                        max_tnm=3,
                                        term_count=5000).search()
    elif method == 'SA':
        method_name = 'Simulated Annealing'
        best_sol, best_cost, data = SA1.SA1(dim, distmat,
                                          term_count_1=200,  # inner loop termination flag
                                          term_count_2=200,  # outer loop termination flag
                                          t_0=1200,  # starting temperature, calculated by init_temp.py
                                          alpha=0.9  # cooling parameter
                                          )
    elif method == 'ISA-2-opt':
        method_name = 'Improved Simulated Annealing - 2-opt Operator'
        best_sol, best_cost, data = SA2.SA2(dim, distmat,
                                          term_count_1=200,  # inner loop termination flag
                                          term_count_2=200,  # outer loop termination flag
                                          t_0=1200,  # starting temperature, calculated by init_temp.py
                                          alpha=0.9  # cooling parameter
                                          )
    elif method == 'ISA-LOOP':
        method_name = 'Improved Simulated Annealing - Loop Improvement'
        best_sol, best_cost, data = SA3.SA3(dim, distmat,
                                          term_count_1=100,  # inner loop termination flag
                                          term_count_2=100,  # outer loop termination flag
                                          t_0=1200,  # starting temperature, calculated by init_temp.py
                                          alpha=0.9  # cooling parameter
                                          )
    elif method == 'ISA-MM':
        method_name = 'Improved Simulated Annealing - Multiple Mutation'
        best_sol, best_cost, data = SA4.SA4(dim, distmat,
                                          term_count_1=100,  # inner loop termination flag
                                          term_count_2=100,  # outer loop termination flag
                                          t_0=1200,  # starting temperature, calculated by init_temp.py
                                          alpha=0.9  # cooling parameter
                                          )
   
    elif method == 'ISA-TP':
        method_name = 'Improved Simulated Annealing - Tuning Parameters'
        best_sol, best_cost, data = SA5.SA5(dim, distmat,
                                          max_tnm=50,  # how many candidates picked in tournament selection
                                          term_count_1=100,  # inner loop termination flag
                                          term_count_2=100,  # outer loop termination flag
                                          t_0=1200,  # starting temperature, calculated by init_temp.py
                                          alpha=0.9  # cooling parameter
                                          )
    elif method == 'ISA-OIS':
        method_name = 'Improved Simulated Annealing - Optimized Initial Solution'
        best_sol, best_cost, data = SA6.SA6(dim, pos, distmat,
                                          max_tnm=50,  # how many candidates picked in tournament selection
                                          term_count_1=100,  # inner loop termination flag
                                          term_count_2=100,  # outer loop termination flag
                                          t_0=1200,  # starting temperature, calculated by init_temp.py
                                          alpha=0.9  # cooling parameter
                                          )
    end = time.time()
    result['time'][_] = end - start
    result['cost'][_] = best_cost
    if best_cost < result['best_cost']:
        result['best_sol'] = best_sol
        result['best_cost'] = best_cost
        result['best_gap'] = best_cost / opt_cost - 1
    plt.plot(range(len(data['cost'])), data['cost'], color='b', alpha=math.pow(num_tests, -0.75))
    plt.plot(range(len(data['cost'])), data['best_cost'], color='r', alpha=math.pow(num_tests, -0.75))

plt.title('Solving TSP with {}'.format(method_name))
plt.xlabel('Number of Iteration')
plt.ylabel('Cost')
plt.savefig('results/{}.png'.format(method))

# print results
result['avg_cost'] = np.mean(result['cost'])
result['avg_gap'] = result['avg_cost'] / opt_cost - 1
result['worst_cost'] = np.max(result['cost'])
result['worst_gap'] = result['worst_cost'] / opt_cost - 1
result['cost_std'] = np.std(result['cost'])
result['cost_std_gap'] = result['cost_std'] / opt_cost
result['avg_time'] = np.mean(result['time'])
result['time_std'] = np.std(result['time'])
del result['best_sol']
del result['cost']
del result['time']
pprint(result)
