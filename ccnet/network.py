import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import repeat, combinations
import time

import networkx as nx

from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors


def network(data, method='nen', k=4, T=0.62, metric='euclidean', weighted=False,
    addknn=0, connected=False, verbose=0):
    """
    Construct a network to approximate the manifold in which the data resides.
    Input:
        data -          n*m np.array
        method -        string, 'nen' - non-uniform epsilon neighborhood(default).
                        'fen' - fixed epsilon neighborhood. 
                        'knn' - k nearest neighbors.

    """
    knn = k
    time = T
    
    n = data.shape[0]
    
    if verbose==1:
        print(' - calculating distance matrix ...')
    # distance matrix
    matrix_dist = squareform(pdist(data, metric=metric))
    
    if verbose==1:
        print(' - calculating growth rates matrix ...')
    # calculate growth rates matrix
    matrix_nn = np.sort(matrix_dist)    # nearest neighbors matrix
    if knn==0:
        growth_rates = np.ones((1, n))
    else:
        growth_rates = np.mean(matrix_nn[:, range(1, knn+1)], axis=-1)
        growth_rates = np.reshape(growth_rates, (1,n))
    
    if verbose==1:
        print(' - calculating times ...')
    # calculate the time at which point pairs connect
    matrix_grow = growth_rates + np.transpose(growth_rates)
    times = matrix_dist / matrix_grow

    if verbose==1:
        print(' - calculating adm ...')
    # generate unweighted adjacency matrix
    adm = np.zeros((n,n), dtype=np.int8)
    adm[times<=time] = 1
    np.fill_diagonal(adm, 0)
    

    # NEN + KNN
    if addknn>0:
        if verbose==1:
            print(' - adding knn network ...')
        k = addknn
        if k > n-1:
            raise Exception('k exceeds n')
        # 计算距离，寻找近邻
        if metric == 'euclidean':
            nbrs = NearestNeighbors(n_neighbors=k+1).fit(data)
            distances, indices = nbrs.kneighbors(data)
        else:
            indices = np.argsort(matrix_dist)
        # 构造邻接矩阵
        for i  in range(n):
            adm[np.ix_([i], indices[i,1:k+1])] = 1
            adm[np.ix_(indices[i,1:k+1], [i])] = 1

                
    # 保持整个网络连通
    if connected:
        bins = connected_components(adm)
        
        if verbose==1:
            if len(bins)>1:
                print(' - network have', len(bins), 'components')
            else:
                print(' network is connected')

        # 错误：这里有错，每次循环，bins的个数应该减一
        while len(bins)>1:
            if verbose==1:
                print('\r - connecting', len(bins), 'compoentes', flush=True)
            
            min_edge = find_min_edge(bins, matrix_dist)
            adm[min_edge[0], min_edge[1]] = 1
            adm[min_edge[1], min_edge[0]] = 1
            if verbose==1:
                print('\r - generating the shortest edge', min_edge)
            
            bins = connected_components(adm)
                
    if weighted:
        wadm = adm * times
        return wadm
    else:
        return adm


def jaccard_net(net):
    """由01网络生成加权网络，可以用组合进行优化"""
    # neighbors contains the i's neighbors, which contains i itself
    
    n = net.shape[0]
    
    # 计算邻居
    neighbors=[set(np.append(np.flatnonzero(net[i]),i).flatten()) for i in range(n)]
    
    # 计算加权网络
    wnet = np.zeros((n,n))
    idx = np.nonzero(net)
    ws = list(map(jaccard_coe, repeat(neighbors), idx[0], idx[1]))
    for i in range(idx[0].shape[0]):
        wnet[idx[0][i], idx[1][i]] = ws[i]
    return wnet

def jaccard_coe(neighbors, i, j):
    """令邻接矩阵的i,j为jaccard coe"""
    
    A, B = neighbors[i], neighbors[j]
    jc = len(A.intersection(B))
    jc /= len(A) + len(B) - jc
    return jc


def connected_components(A, verbose=0):
    """
    Compute the connected components of network A
    Input:
        A  - n*n np.array(), adjacency matrix of the network
    Output:
        bins - a dictionary, key is number represent id of bin,
                and value is list containing each connected components
    """
    
    G = nx.from_numpy_matrix(A)
    
    bins = {}
    ind = 0
    for c in nx.connected_components(G):
        bins[ind] = list(c)
        ind += 1
        
    return bins

def min_dist(bin0, bin1, dm):
    """计算两个桶之间的最小距离
        bin0: the first bin
        bin1: the second bin
        dm: distance matrix
    return 
        edge: (node in bin0, node in bin1)
        d: min dist between two bins"""
    
    edge = [bin0[0], bin1[0]]
    d = dm[edge[0], edge[1]]
    
    for i in bin0:
        for j in bin1:
            if dm[i,j] < d:
                edge = [i,j]
                d = dm[i,j]
    return edge, d

def find_min_edge(bins, dm, verbose=0):
    """寻找连通分支之间最短的边
    parameters:
        adm: adjacency matrix
        dm: distance matrix
    return:
        edge: closest edge among connected components
    """
    
#     bins = connected_components(adm, verbose=verbose)
    
    # 计算各桶之间的最短边和距离
    min_edges = {}
    for i,j in combinations(range(len(bins)), 2):
        min_edges[(i,j)] = min_dist(bins[i], bins[j], dm)

    # 寻找最短的边和距离
    edge, d = min_edges[(0,1)]
    for ve, vd in min_edges.values():
        if vd < d:
            edge, d = ve, vd
    return edge