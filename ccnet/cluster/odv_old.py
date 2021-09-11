import numpy as np
from random import shuffle
import copy

class Partition:
    """partition
    labels key-节点， value为社区
    coms key-社区， value为节点
    """
    labels = {}
    coms = {}
    
    def __init__(self, n):
        """初始化partition"""
        self.labels = {}
        self.coms = {}
        if type(n) is not int:
            raise Exception('n must be an interger!')
            
        for i in range(n):
            self.labels[i] = i
            self.coms[i] = [i]
            
    def node_to_com(self, node, com):
        """move node to community com"""
        self.coms[self.labels[node]].remove(node)
        self.coms[com].append(node)
        self.labels[node] = com
        
    def flatten(self, par):
        """flatten partition"""
        new_coms = {}
        for i, com in self.coms.items():
            new_coms[i] = []
            for node in com:
                new_coms[i] = new_coms[i] + par.get_com(node)
                
        self.coms = new_coms
        self.labels = {}
        for i, value in self.coms.items():
            for node in value:
                self.labels[node] = i
                
        return self
    
    def separate_bridge(self, com):
        """将桥社区拆开"""
        if sorted(self.coms.keys())!=list(range(len(self.coms))):
            raise Exception('not updated!')
        self.coms[len(self.coms)] = [com[1]]
        self.coms[self.labels[com[0]]].remove(com[1])
        self.labels[com[1]] = len(self.coms) - 1
        
    def update(self):
        """delete empty communities, renumber"""
        # remove empty communities
        keys = [key for key, value in self.coms.items() if not value]
        for key in keys:
            del self.coms[key]
        
        # renumber
        temp = {}
        for i, key in enumerate(sorted(self.coms.keys())):
            temp[i] = self.coms[key]
            for node in temp[i]:
                self.labels[node] = i
            
        # update self.labels
        self.coms = temp
        
    def com_of_node(self, node):
        """返回节点所在的社区"""
        return self.labels[node]
    
    def get_com(self, com):
        """返回整个社区"""
        return self.coms[com]
    
    def get_num(self):
        """返回社区数"""
        return len(self.coms)

    def get_lab(self):
        """返回labels"""
        return [self.labels[i] for i in range(len(self.labels))]
            
    def show(self):
        """show labels and coms"""
        print(' ---- labels and coms --- ')
        print(' - labels = ', self.labels)
        print(' - coms = ', self.coms)
        print(' ------------------------ ')

def dvalue_partition(net, node, partition, verbose=0):
    """当前网络在当前划分的d-value"""
    
    dvalue = 0.
    for value in partition.coms.values():
        dvalue += dvalue_community(net, node, value)
    
    return dvalue

def dvalue_community(net, node, community, verbose=0):
    """当前网络中，某一社区的d-value"""

    if not community:
        return 0.
    
    kin = 0.
    kout = 0.
    weights = 0.
    
    for i in range(len(community)):
        for j in range(len(node)):
            if net[community[i], j]:
                # if node j is in community
                if j in community:
                    kin += net[community[i], j]
                else:
                    kout += net[community[i], j]
        weights += node[community[i]]
        
    return (kin - kout) / weights

def odv(net, ifcheck=True, random=True, verbose=0):
    """对网络进行社团检测"""
    par, dv = communities(net=net, ifcheck=ifcheck, random=random, verbose=verbose)
    return par.get_lab()

def communities(net, ifcheck=True, random=True, verbose=0):
    """对网络进行社团检测
    ifcheck - boolean, True - have check phase, False - doesn't have
    """
    
    n = net.shape[0]
    node_weights = [1 for i in range(n)]
    
    # initialize partition
    partition0 = Partition(n)
    print(' - initialize ...')
    # partition0.show()
    iteration = 0
    print(' - iteration = ', iteration)
    iteration += 1
    d0 = dvalue_partition(net, node_weights, partition0, verbose=verbose)
    print(' - d0 = ', d0)
    partition1 = move(net, node_weights, partition0, random=random, verbose=verbose)
    
    # !!!
    if ifcheck:
        partition1 = check(net, node_weights, partition1, verbose=verbose)
    
    [net1, node1] = aggregate_partition(net, node_weights, partition1, verbose=verbose)
    d1 = dvalue_partition(net, node_weights, partition1, verbose=verbose)
    
    # current partition and D-value
    partition = partition1
    dvalue = d1
    
    while d1 > d0:
        d0 = d1
        net0 = net1
        node0 = node1
        
        print(' - iteration = ', iteration)
        iteration += 1
        # initialize partition
        # partition0.show()
        partition0 = Partition(len(node0))
        #partition0 = singleton_partition(len(node0))
        
        # print(' - net0 = ', net0)
        # print(' - node0 = ', node0)
        # partition0.show()
        
        # move -> aggregate
        partition1 = move(net0, node0, partition0, random=random, verbose=verbose)
        if ifcheck:
            partition1 = check(net0, node0, partition1, verbose=verbose)
        [net1, node1] = aggregate_partition(net0, node0, partition1)
        d1 = dvalue_partition(net0, node0, partition1)
        
        # current partition and D-value
        partition = partition1.flatten(partition)
        dvalue = d1
        
        # partition.show()
    return partition, dvalue

def move(net, node, partition, random=True, verbose=0):
    """一直调用move_nodes, 直到d-value不再增加为止"""
    # print(' - now in move() ... ')
    d0 = dvalue_partition(net, node, partition, verbose=verbose)
    print(' - 0, dv = ', d0)
    partition = move_nodes(net, node, partition, random=True, verbose=verbose)
    # partition.show()
    d1 = dvalue_partition(net, node, partition, verbose=verbose)
    print(' - 1, dv = ', d1)

    flag = 1
    while d1 > d0:
        d0 = d1
        partition = move_nodes(net, node, partition,random=True, verbose=verbose)
        # partition.show()
        d1 = dvalue_partition(net, node, partition, verbose=verbose)
        print(' - ', flag, ', dv = ', d1)
        flag += 1

    return partition
    
def move_nodes(net, node, partition, random=True, verbose=0):
    """移动网络中的每个节点"""
    # 计算节点个数
    n = len(node)
    
    # 节点访问顺序
    nodelist = [i for i in range(n)]
    if random:
        shuffle(nodelist)
    
    # 移动每个节点，this为当前节点
    for i in range(n):
        print('\r  - i =', i, end='')
        this = nodelist[i]

        # 计算节点的所有邻居
        neighbors = [i for i in range(n) if net[this, i]>0 and this != i]
        if random:
            shuffle(neighbors)

            
        # 如果邻居为空，则继续下一次移动
        if not neighbors:
            print('  - neighbor is empty')
            continue
            
        # 计算当前节点所在的社区，和当前社区的dvalue
        comm0 = partition.com_of_node(this)
        dvalue_comm0 = dvalue_community(net, node, partition.get_com(comm0), verbose=verbose)
                  
        best_com = comm0
        max_inc = 0
        
        # 对所有的邻居进行遍历，neighbor为当前邻居
        for neighbor in neighbors:
            # 计算当前节点移动到当前邻居所在社区后的d-value
            
            # 计算当前邻居所在的社区
            comm1 = partition.com_of_node(neighbor)
            
            if best_com==comm1:
                continue
            else:
                # 计算移动前，两个社区的D-value之和
                d0 = dvalue_comm0 + dvalue_community(net, node, partition.get_com(comm1), verbose=verbose)
                # 移动this
                partition.node_to_com(this, comm1)
                # 计算移动后两个社区的d-value之和
                d1 = dvalue_community(net, node, partition.get_com(comm0)) + dvalue_community(net, node, partition.get_com(comm1))
                # 将this归位
                partition.node_to_com(this, comm0)

                if d1-d0 > max_inc:
                    max_inc = d1-d0
                    best_com = comm1
        # 移动this 到best_com
        if best_com!=comm0:
            partition.node_to_com(this, best_com)
        # partition.show()
    
    print('')
    partition.update()
    return partition
            

def community_bridges(net, node, partition, verbose=0):
    """检查是否存在负d-value且具有两个节点的社区，如果存在，返回所有该社区序号，否在，返回空"""
    result = {}
    for i in range(partition.get_num()):
        com = partition.get_com(i)
        dvalue = dvalue_community(net, node, com, verbose=verbose)
        if len(com)==2 and dvalue<0:
            result[i] = dvalue
    # 根据dvalue对桥社区排序，只返回社区序号
    resultlist = sorted(result.items(), key=lambda item:item[1])
    return [resultlist[i][0] for i in range(len(resultlist))]

def isacommunity(a, partition):
    """判断a是否是一个社区，如果是，则返回a在partition的键，如果不是，则返回-1.
    优化：set(a)==set(partition.get_com(i))可写为 a==partition.get_com(i),
    因为在check过程中，社区节点的顺序保持不变
    """
    result = -1
    flag = 0
    for i in range(partition.get_num()):
        if set(a)==set(partition.get_com(i)):
            result = i
            flag += 1
    if flag > 1:
        raise Exception('Have the same communities!')
        
    return result

def max_com(node1, node2, net, node_weights, par, verbose=0):
    """
    找到移动node1 最大的社区，且node2 不能作为node1 的邻居
    """
    
    n = net.shape[0]
    
    # node1's neighbors, which doesn't comtain node1 and node2
    neighbors = []
    for i in range(n):
        if net[node1, i]>0 and i!=node1 and i!=node2:
            neighbors.append(i)
         
    # 计算当前节点所在社区，和该社区的dvalue
    com1 = par.com_of_node(node1)
    dv_com1 = dvalue_community(net, node_weights, par.get_com(com1), verbose=verbose)
    
    best_com = -1
    max_inc = 0
    
    # 对所有的邻居进行遍历，neighbor为当前的邻居
    for neighbor in neighbors:
        # 计算当前邻居所在的社区
        com2 = par.com_of_node(neighbor)
        d0 = dv_com1 + dvalue_community(net, node_weights, par.get_com(com2), verbose=verbose)
        # 移动node1
        par.node_to_com(node1, com2)
        # 计算移动后两个社区的d-value之和
        d1 = dvalue_community(net, node_weights, par.get_com(com1)) +\
            dvalue_community(net, node_weights, par.get_com(com2))
        # 将node1归位
        par.node_to_com(node1, com1)
        if d1-d0 > max_inc:
            max_inc = d1-d0
            best_com = com2
            
    return best_com

def check(net, node, partition, verbose=0):
    """improve bridge communities"""
    
    par = copy.deepcopy(partition)
    
    # check if have bridge communities
    bcs = community_bridges(net, node, partition, verbose=verbose)
#     print(bcs)
    
    # if bcs is empty, return original partition directly. otherwise, import them
    if not bcs:
        return partition
    else:
        for i in range(len(bcs)):
            # 这里需要检查原先的社区，在新的划分中，是否还是一个社区。因为可能由于之前桥社区
            # 的拆解和合并，使得其它桥社区变成了非桥社区
            # isacommunity() 返回社区在划分中的键，如果划分中不存在该键，则返回-1
            com = partition.get_com(bcs[i])
            c_bridge = isacommunity(com, par)
            
            if c_bridge>-1:
                par1 = copy.deepcopy(par)
                
                # 两个桥节点作为一个社区的D-value
                d1 = dvalue_partition(net, node, par, verbose=verbose)
                
                # ！！！ 这里有改动
                
                # step 1: 将社区拆开；将社区的第二个节点放在总划分的最后，第一个节点仍然留在原社区
                par1.separate_bridge(com)

                # par1.show()

                node1 = com[0]
                node2 = com[1]
        
                par1.com_of_node(node1)
                par1.com_of_node(node2)
                
#                 print(c_node1)
#                 print(c_node2)
                
                # 两个桥节点
#                 node1 = par1[c_node1][0]
#                 node2 = par1[c_node2][0]
                
                # 两个桥节点拆开，分别作为一个社区的 d-value
                d0 = dvalue_partition(net, node, par1)
                
                # step 2: 找使移动桥节点1最大的社区
                com1 = max_com(node1, node2, net, node, par1, verbose=verbose)
                
                # step 3: 找使移动节点2最大的社区
                com2 = max_com(node2, node1, net, node, par1, verbose=verbose)
                
                # step 4: 将node1, node2 分别移入社区com1, com2，然后移除旧社区
                par1.node_to_com(node1, com1)
                par1.node_to_com(node2, com2)
#                 par1[com1].append(node1)
#                 par1[com2].append(node2)
                
#                 del par1[c_node1]
#                 del par1[c_node2]
                
                #!!!这里可以进行优化 
                
                # 删掉空社区，并重新编号
#                 par1 = update_partition(par1, verbose=verbose)
                par1.update()
                
                # 将bridge 拆分且移动后的dvalue
                d2 = dvalue_partition(net, node, par1)
                
                if d2 > d1:
                    par = par1
                    
            else:
                continue
    return par
    

def aggregate_partition(net, node, par, verbose=0):
    """将划分聚合成一个新网络"""
    n1 = par.get_num()
    net1 = np.zeros((n1, n1))
    node1 = [0 for i in range(n1)]
    
    # 计算新节点的权重
    for i in range(n1):
        for j in par.get_com(i):
            node1[i] += node[j]
    
    # 计算新的连边
    n = len(node)
    for i in range(n):
        # diagonal elements
        net1[par.com_of_node(i), par.com_of_node(i)] += net[i,i]
        # non-diagonal elements
        for j in range(i+1, n):
            if net[i, j] > 0:
                net1[par.com_of_node(i), par.com_of_node(j)] += net[i, j]
                net1[par.com_of_node(j), par.com_of_node(i)] += net[j, i]
    return net1, node1
        