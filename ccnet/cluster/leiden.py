import community as community_louvain
import networkx as nx

import leidenalg as la
import igraph as ig

import numpy as np

def par2par_louvain(par):
    """转化为louvain可以使用的划分"""
    # 计算节点数
    partition = {}
    for i in range(len(par)):
        for j in range(len(par[i])):
            partition[par[i][j]] = i
    return partition
    
def par2labels(par):
    """划分转化为标签"""
    # 计算节点数
    n = 0
    for i in range(len(par)):
        n += len(par[i])
    # 初始化标签,z
    labels = [-1 for i in range(n)]
    for i in range(len(par)):
        for j in range(len(par[i])):
            labels[par[i][j]] = i
    return labels

def leiden(net, weighted=False, quantity='modularity', resolution=0.05, n_iters=1):
    """"""
    G_nx = nx.from_numpy_array(net)
    g = ig.Graph.from_networkx(G_nx)
    if quantity=='modularity':
        par = la.find_partition(g, la.ModularityVertexPartition)
    elif quantity=='CPM':
        par = la.find_partition(g, la.CPMVertexPartition, resolution_parameter=resolution)
    else:
        raise Exception('wrong quantity')
    m = community_louvain.modularity(par2par_louvain(par), G_nx)
    
    i = 1
    print()
    while i < n_iters:
        # par0 = la.find_partition(g, la.ModularityVertexPartition)
        if quantity=='modularity':
            par0 = la.find_partition(g, la.ModularityVertexPartition)
        elif quantity=='CPM':
            par0 = la.find_partition(g, la.CPMVertexPartition, resolution_parameter=resolution)
        else:
            raise Exception('wrong quantity')
        m0 = community_louvain.modularity(par2par_louvain(par0), G_nx)
        print('\r     {} of {}, modularity = {:.4f}'.format(i, n_iters, m), sep='', end='')
        i += 1
        if m0 > m:
            par = par0
            m = m0
            
    return par2labels(par)

# def leiden(net, weighted=False, quantity='modularity', resolution=0.05, n_iters=1):
#     """"""
#     G_nx = nx.from_numpy_array(net)
#     g = ig.Graph.from_networkx(G_nx)
#     if quantity=='modularity':
#         par = la.find_partition(g, la.ModularityVertexPartition)
#     elif quantity=='CPM':
#         par = la.find_partition(G, la.CPMVertexPartition, resolution_parameter=resolution)
#     else:
#         raise Exception('wrong quantity')
#     m = community_louvain.modularity(par2par_louvain(par), G_nx)
    
#     i = 1
#     print()
#     while i < n_iters:
#         par0 = la.find_partition(g, la.ModularityVertexPartition)
#         m0 = community_louvain.modularity(par2par_louvain(par0), G_nx)
#         print('\r     {} of {}, modularity = {:.4f}'.format(i, n_iters, m), sep='', end='')
#         i += 1
#         if m0 > m:
#             par = par0
#             m = m0
            
#     return par2labels(par)