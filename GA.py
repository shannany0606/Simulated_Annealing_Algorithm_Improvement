import numpy as np
from common import get_cost
import random
from collections import deque
from common import Init

class GA:
    def __init__(self, pos, adj_mat, n_pop, r_cross, r_mut, max_tnm, term_count):
        self.distmat=adj_mat  # 距离矩阵
        self.dim=adj_mat.shape[0]  # 维度
        self.og=n_pop  # 种群数目
        self.group=[]  # 种群
        self.path=None
        self.length=0
        self.lengths=[]
        self.pc=r_cross  # 交叉概率
        self.pm=r_mut  # 变异概率
        self.mc=self.og  # 种群中至少个体数
        self.points=pos
        self.max_tnm=max_tnm
        self.term_count=term_count
 
    def tnm_selection(self, pop, costs, max_tnm):  # tournament selection
        selection_ix = np.random.randint(len(pop))  # [0, len(pop)-1]
        for _ in np.random.randint(0, len(pop), max_tnm - 1):  # [0, len(pop)-1]
            # check if better (e.g. perform a tournament)
            if costs[_] < costs[selection_ix]:
                selection_ix = _
        return pop[selection_ix]

    def rw_selection(self, pop, costs, _):  # roulette-wheel selection
        costs = [max(costs) - x for x in costs]  # avoid negative scores
        sum_score = sum(costs)
        return pop[np.random.choice(len(pop), p=[x / sum_score for x in costs])]

    def elt_selection(self, pop, costs, _):  # elitism
        return pop[np.argmin(costs)]

    def crossOver_OX(self):
        temp=self.og
        for i in range(temp // 2):
            if np.random.random()>self.pc:  # 判断是否交叉
                continue
            parent1 = self.group[i]
            parent2 = self.group[i + temp // 2]
            oops = np.random.randint(1, self.dim)
            child1, child2 = list(parent1[:oops].copy()), list(parent2[:oops].copy())
            for j in range(len(parent1)):  # 解决冲突
                if parent1[j] not in child2:
                    child2.append(parent1[j])
                if parent2[j] not in child1:
                    child1.append(parent2[j])
            self.group.append(child1.copy())
            self.group.append(child2.copy())
            self.og+=2

    def crossOver_Heuristic(self):
        temp=self.og
        for i in range(temp // 2):
            if np.random.random()>self.pc:  # 判断是否交叉
                continue
            parent1 = self.group[i]
            parent2 = self.group[i + temp // 2]
            st = np.random.randint(1, self.dim)
            child1 = [st]
            child2 = [st]
            n = len(parent1)
            for j in range(n - 1):
                nw = child1[-1]
                
                pos = parent1.index(nw)
                while True:
                    pos = (pos + 1) % n
                    nxt1 = parent1[pos]
                    if nxt1 not in child1:break
                    #print(len(child1))

                pos = parent2.index(nw)
                while True:
                    pos = (pos + 1) % n
                    nxt2 = parent2[pos]
                    if nxt2 not in child1:break

                if self.distmat[nw][nxt1] < self.distmat[nw][nxt2]:
                    child1.append(nxt1)
                else:child1.append(nxt2)
            for j in range(n - 1):
                nw = child2[-1]

                pos = parent1.index(nw)
                while True:
                    pos = pos - 1
                    if pos < 0: pos += n
                    pre1 = parent1[pos]
                    if pre1 not in child2:break

                pos = parent2.index(nw)
                while True:
                    pos = pos - 1
                    if pos < 0: pos += n
                    pre2 = parent2[pos]
                    if pre2 not in child2:break

                if self.distmat[nw][pre1] < self.distmat[nw][pre2]:
                    child2.append(pre1)
                else:child2.append(pre2)
            self.group.append(child1.copy())
            self.group.append(child2.copy())
            self.og+=2

    def mutate_swap(self):
        for i in range(self.og):
            if np.random.random()<self.pm:
                random_site0 = np.random.randint(0, self.dim-1)
                random_site1 = np.random.randint(random_site0,self.dim)
                # 交换两个城市
                self.group[i][random_site0], self.group[i][random_site1] = self.group[i][random_site1], self.group[i][random_site0]

    def mutate_inversion(self):
        for i in range(self.og):
            if np.random.random()<self.pm:
                random_site0 = np.random.randint(0, self.dim-1)
                random_site1 = np.random.randint(random_site0,self.dim)
                # 交换两个城市并将之间的城市倒置
                while random_site0 < random_site1:
                    self.group[i][random_site0],self.group[i][random_site1]=self.group[i][random_site1],self.group[i][random_site0]
                    random_site0+=1
                    random_site1-=1

    #中心对换变异
    def mutate_CIM(self):
        for i in range(self.og):
            if random.random() < self.pm:
                Ct = random.randint(0, self.dim-1)
                self.group[i] = self.group[i][:Ct][::-1] + self.group[i][Ct:][::-1]

    # 选择个体轮盘赌策略
    def select_roulette(self):
        fits=self.groupFitness()
        temp_group=np.array(self.group)
        new_group_index=np.random.choice(range(self.og),size=self.mc,replace=True,p=fits/fits.sum())
        # 轮盘赌选出相应个体的索引
        new_group=temp_group[new_group_index]  # 选出新个体
        self.group=new_group.tolist()
        self.og=self.mc

    # 精英保留策略
    def select_optimal(self):
        fits=self.groupFitness()
        temp_group=np.array(self.group)
        new_group_index=np.random.choice(self.og,size=self.mc,replace=True,p=fits/fits.sum())
        # 先进行轮盘赌策略筛选个体
        new_group=temp_group[new_group_index]
        new_fits=fits[new_group_index]
        max_fit=fits.argmax()
        min_fit=new_fits.argmin()
        new_group[min_fit]=temp_group[max_fit]  # 最优个体替换
        self.group=new_group.tolist()
        self.og=self.mc

    # 个体适应值函数
    def fitness(self,status):
        return 10000/get_cost(self.dim, self.distmat, status)

    # 获得种群中的所有个体的适应值和总和
    def groupFitness(self):
        fits=np.zeros(self.og)
        for i in range(self.og):
            fits[i]=self.fitness(self.group[i])
            pass
        return fits

    def search(self):
        #print(self.points)
        #self.group = [random.sample(range(self.dim), self.dim) for _ in range(self.og)]
        self.group = Init(self.points, self.dim, self.og, self.distmat).newCreate()
        #print(self.group)
        best_sol, best_cost = self.group[0], get_cost(self.dim, self.distmat, self.group[0])  # randomly initialize best!
        data = {'cost': deque([]), 'best_cost': deque([])}  # cost means avg_cost
        count = 0
        while True:
            costs = [get_cost(self.dim, self.distmat, _) for _ in self.group]
            last_best_cost = best_cost
            for i in range(self.og):
                if costs[i] < best_cost:
                    best_sol, best_cost = self.group[i], costs[i]
            if last_best_cost == best_cost:  # best_cost not change
                count += 1
            else:  # best_cost change
                count = 0
            data['cost'].append(np.mean(costs))
            data['best_cost'].append(best_cost)
            if count > self.term_count:
                return best_sol, best_cost, data
            # create the next generation
            self.crossOver_OX()
            #self.crossOver_Heuristic()
            #self.mutate_inversion()
            self.mutate_CIM()
            self.select_optimal()