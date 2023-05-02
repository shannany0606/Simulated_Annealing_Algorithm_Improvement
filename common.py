import random
import math
def get_cost(n, adj_mat, sol):
    """
    :param n: number of vertices, e.g. 2
    :param adj_mat: adjacency matrix, e.g. [[0,1], [1,0]]
    :param sol: solution, e.g. [1,0]
    """
    return sum([adj_mat[sol[_]][sol[(_ + 1) % n]] for _ in range(n)])

def get_new_sol_swap(sol, i, j):
    new_sol = sol.copy()
    new_sol[i], new_sol[j] = new_sol[j], new_sol[i]
    return new_sol

def get_delta_swap(n, adj_mat, sol, i, j):
    # bef: [..., i-1, i, i+1, ..., j-1, j, j+1] / [...,i-1, i, j, j+1, ...]
    # aft: [..., i-1, j, i+1, ..., j-1, i, j+1] / [...,i-1, j, i, j+1, ...]
    # the latter case, 2 * adj_mat(i, j) is extra deducted!
    delta = adj_mat[sol[i - 1]][sol[j]] + adj_mat[sol[j]][sol[(i + 1) % n]] + \
            adj_mat[sol[j - 1]][sol[i]] + adj_mat[sol[i]][sol[(j + 1) % n]] - \
            adj_mat[sol[i - 1]][sol[i]] - adj_mat[sol[i]][sol[(i + 1) % n]] - \
            adj_mat[sol[j - 1]][sol[j]] - adj_mat[sol[j]][sol[(j + 1) % n]]
    if j - i == 1 or i == 0 and j == n - 1:
        delta += 2 * adj_mat[sol[i]][sol[j]]  # symmetrical TSP
    return delta

def get_new_sol_2opt(sol, i, j):
    new_sol = sol.copy()
    new_sol[i:j+1] = new_sol[i:j+1][::-1]  # notice index + 1 !
    return new_sol

def get_delta_2opt(n, adj_mat, sol, i, j):
    # bef: [..., i-1, i, i+1, ..., j-1, j, j+1] / [...,i-1, i, j, j+1, ...] / [i, i+1, ..., j-1, j]
    # aft: [..., i-1, j, j-1, ..., i+1, i, j+1] / [...,i-1, j, i, j+1, ...] / [j, i+1, ..., j-1, i]
    # the latter case, 2 * adj_mat(i, j) is extra deducted!
    delta = adj_mat[sol[i - 1]][sol[j]] + adj_mat[sol[i]][sol[(j + 1) % n]] - \
            adj_mat[sol[i - 1]][sol[i]] - adj_mat[sol[j]][sol[(j + 1) % n]]
    if i == 0 and j == n - 1:  # the first two value == 0, while others < 0
        delta = 0
    return delta

def get_new_sol_move(sol, i, j):
    new_sol = sol.copy()
    new_sol[i:j+1] = new_sol[i:i+1] + new_sol[j:j+1] + new_sol[i+1:j]  # notice index + 1 !
    return new_sol

def get_delta_move(n, adj_mat, sol, i, j):
    # bef: [..., i-1, i, i+1, ..., j-1, j, j+1] 
    # aft: [..., i-1, i, j, i+1, ..., j-1,j+1] 
    # the latter case, 2 * adj_mat(i, j) is extra deducted!
    delta = adj_mat[sol[i]][sol[j]] + adj_mat[sol[j]][sol[(i + 1) % n]] + adj_mat[sol[j - 1]][sol[(j + 1) % n]] - \
            adj_mat[sol[i]][sol[(i + 1) % n]] - adj_mat[sol[j - 1]][sol[j]] - adj_mat[sol[j]][sol[(j + 1) % n]]
    if i == n - 1 and j == 0:  # the first two value == 0, while others < 0
        delta = 0
    return delta

def get_new_sol_CIM(sol, i, j):
    new_sol = sol.copy()
    new_sol = new_sol[:i][::-1] + new_sol[i:][::-1]
    return new_sol

def get_delta_CIM(n, adj_mat, sol, i, j):
    # bef: [0 , 1 , ... , i-1 , i , i+1 , ... , n-1] 
    # aft: [i-1 , i-2 , ... , 0 , i , n-1 , n-2 , ... , i+1] 
    # the latter case, 2 * adj_mat(i, j) is extra deducted!
    delta = adj_mat[sol[0]][sol[i]] + adj_mat[sol[i]][sol[n - 1]] + adj_mat[sol[i - 1]][sol[(i + 1) % n]] - \
            adj_mat[sol[i - 1]][sol[i]] - adj_mat[sol[i]][sol[(i + 1) % n]] - adj_mat[sol[0]][sol[n - 1]]
    if i == 0 and i == n - 1:  # the first two value == 0, while others < 0
        delta = 0
    return delta

#种群初始化
class Init:
    def __init__(self, points, dim, og, distmat):
        self.points = points
        self.dim = dim
        self.og = og
        self.distmat = distmat
        self.group = []
        
    # 计算两个向量之间的叉积。返回三点之间的关系：    
    def ccw(self,a, b, c):
        return ((b[1] - a[1]) * (c[0] - b[0]))   -    ((c[1] - b[1]) * (b[0] - a[0]) )

# 分别求出后面n-1个点与出发点的斜率，借助sorted，按斜率完成从小到大排序
    def compute(self,next):
        start = self.points[0]  # 第一个点
        if start[0] == next[0]: 
            return 99999
        slope = (start[1] - next[1]) / (start[0] - next[0])  
        return slope

    def Graham_Scan(self,points):
    # # 找到最左边且最下面的点作为出发点，和第一位互换
        Min=999999
        for i in range(len(points)):
        # 寻找最左边的点
            if points[i][0]<Min:
                Min = points[i][0]
                index = i
        # 如果同在最左边，可取y值更小的
            elif points[i][0]==Min:
                if points[i][1]<=points[index][1]:
                    Min = points[i][0]
                    index = i
        # 和第一位互换位置
        temp = points[0]
        points[0] = points[index]
        points[index] = temp
        # 排序：从第二个元素开始，按与第一个元素的斜率排序
        points = points[:1] + sorted(points[1:], key=self.compute)   # 前半部分是出发点；后半部分是经过按斜率排序之后的n-1个坐标点
        #注意： “+”是拼接的含义，不是数值相加
        # 用列表模拟一个栈。（最先加入的是前两个点，前两次while必定不成立，从而将点加进去）
        convex_hull = []
        for p in points:
            while len(convex_hull) > 1 and self.ccw(convex_hull[-2], convex_hull[-1], p) >= 0:
                convex_hull.pop()
            convex_hull.append(p)
        person=[x[2] for x in convex_hull]
        return person

    #优化后的初始种群产生方法
    def newCreate(self):
        stp=self.Graham_Scan(self.points)
        origin=[i for i in range(self.dim)]
        for x in stp:
                origin.remove(x)
        sto=origin
        for i in range(self.og):
            person=stp.copy()#不加.copy()是赋了指针！！！
            #print("st",person)
            origin=sto.copy()
            toto,totp=len(sto),len(stp)
            #print(toto,totp)
            #print(origin)
            while toto>0:
                x=origin[random.randint(0,toto-1)]
                mn,posmn=999999,0
                for j in range(totp):
                    if j<totp-1:Dis=self.distmat[person[j],x]+self.distmat[x,person[j+1]]-self.distmat[person[j],person[j+1]]
                    else:Dis=self.distmat[person[j],x]+self.distmat[x,person[0]]-self.distmat[person[j],person[0]]
                    if Dis<mn:
                        mn=Dis
                        posmn=j
                person.insert(posmn+1,x)
                totp+=1
                origin.remove(x)
                toto-=1
            self.group.append(person)  # 产生随机个体
            #print(len(person),self.dim)
            #print(person)
        if self.og == 1:return self.group[0]
        return self.group

#Tuning parameters
def tnm_selection(n, adj_mat, sol, max_tnm, mut_md):
    """
    :param n: number of vertices
    :param adj_mat: adjacency matrix
    :param sol: solution where the neighbours are chosen from
    :param max_tnm: how many candidates picked in tournament selection
    :param mut_md: [get_sol, get delta], method of mutation, e.g. swap, 2-opt
    """

    get_new_sol = mut_md[0]
    get_delta = mut_md[1]

    cost = get_cost(n, adj_mat, sol)

    best_delta  = math.inf
    best_i = best_j = -1

    for _ in range(max_tnm):
        i, j = random.sample(range(n), 2)  # randomly select two indexes
        i, j = (i, j) if i < j else (j, i)  # let i < j
        delta = get_delta(n, adj_mat, sol, i, j)
        if delta < best_delta:
            best_delta = delta
            best_i = i
            best_j = j
    new_sol = get_new_sol(sol, best_i, best_j)
    new_cost = cost + best_delta
    # assert abs(new_cost - get_cost(n, adj_mat, new_sol)) < 1e-9, 'new_sol does not match new_cost'
    return new_sol, new_cost

def Search(n, adj_mat, sol, mut_md):
    """
    :param n: number of vertices
    :param adj_mat: adjacency matrix
    :param sol: solution where the neighbours are chosen from
    :param mut_md: [get_sol, get delta], method of mutation, e.g. swap, 2-opt
    """

    get_new_sol = mut_md[0]
    get_delta = mut_md[1]

    cost = get_cost(n, adj_mat, sol)

    i, j = random.sample(range(n), 2)  # randomly select two indexes
    i, j = (i, j) if i < j else (j, i)  # let i < j
    delta = get_delta(n, adj_mat, sol, i, j)

    new_sol = get_new_sol(sol, i, j)
    new_cost = cost + delta
    # assert abs(new_cost - get_cost(n, adj_mat, new_sol)) < 1e-9, 'new_sol does not match new_cost'
    return new_sol, new_cost