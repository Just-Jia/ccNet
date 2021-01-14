"""
对数据进行预处理，建立过滤网络和过滤值下的网络
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform

from .trajectory import connected_components

def network(data, method='nen', knn=3, time=1.5, metric='euclidean', weighted=False, 
    verbose=0):
    """
    Construct a network to approximate the manifold in which the data resides.
    Input:
        data -          n*m np.array
        method -        string, 'nen' - non-uniform epsilon neighborhood(default).
                        'fen' - fixed epsilon neighborhood. 
                        'knn' - k nearest neighbors.

    """
    knn = knn
    time = time
    
    n = data.shape[0]
    
    # distance matrix
    matrix_dist = squareform(pdist(data, metric='euclidean'))

    # calculate growth rates matrix
    matrix_nn = np.sort(matrix_dist)    # nearest neighbors matrix
    if knn==0:
        growth_rates = np.ones((1, n))
    else:
        growth_rates = np.mean(matrix_nn[:, range(1, knn+1)], axis=-1)
        growth_rates = np.reshape(growth_rates, (1,n))

    # calculate the time at which point pairs connect
    matrix_grow = growth_rates + np.transpose(growth_rates)
    times = matrix_dist / matrix_grow

    # adjacency matrix
    if weighted:
        adm = np.zeros((n,n))
        for i in range(n):
            for j in range(i+1,n):
                if times[i,j] < time:
                    adm[i,j] = times[i,j]
                    adm[j,i] = times[i,j]
    else:
        adm = np.zeros((n,n))
        for i in range(n):
            for j in range(i+1,n):
                if times[i,j] < time:
                    adm[i,j] = 1
                    adm[j,i] = 1
    return adm


def filter_network(datafile, max_time=1.5, knn=3):
    """
    Build filter network, actually neighborhood graph.
    input:
        datafile -  filename of data
        max_time -  a number, the maximum growth time of open ball for each 
                    point. default 1.5
        knn -       a number, the nearest neighbor number for calculating the
                    grow rates
    output:
        nodes -     a list, containing n data points.
        edges -     a list, containing the point pairs as edges.
        time_weights -  a list, the same size as 'edges', indicating the of appearance.
    """
    verbose = 0

    data = np.array(pd.read_csv(datafile, header=None))
    if verbose==1:
        plt.scatter(data[:, 0], data[:,1])
        plt.show()

    n = data.shape[0]   # points number

    # initialization
    nodes = [i for i in range(n)]
    edges =[]
    time_weights = []

    if verbose:
        print(' - node = ', nodes)
        print(' - edges = ', edges)
        print(' - time_weights = ', time_weights)

    # distant matrix
    matrix_dist = squareform(pdist(data, metric='euclidean'))

    if verbose:
        print( ' - matrix_dist type = ', matrix_dist.shape, type(matrix_dist))
        # print( ' - matrix_dist = ', matrix_dist)

    # calculate growth rates matrix
    matrix_nn = np.sort(matrix_dist)    # nearest neighbors matrix
    if knn==0:
        growth_rates = np.ones((1, n))
    else:
        growth_rates = np.mean(matrix_nn[:, range(1, knn+1)], axis=-1)
        growth_rates = np.reshape(growth_rates, (1,n))

    if verbose:
        print(' - growth_rates = ', growth_rates.shape, type(growth_rates))

    # grwth rate for point pairs
    matrix_grow = growth_rates + np.transpose(growth_rates)
    times = matrix_dist/matrix_grow

    # calculate edges and its time wights
    for i in range(n):
        for j in range(i+1, n):
            if times[i][j] < max_time:
                edges.append((i, j))
                time_weights.append(times[i][j])

    if verbose:
        print(' - edges = ', len(edges), type(edges))
        print(' - time_weights = ', len(time_weights), type(edges))

    return nodes, edges, time_weights

def fnetwork_to_network(nodes, edges, time_weights, time=1):
    """
    Convert filter network at certain time to adjacency matrix, 
    input:
        nodes -     a list, containing the node
        edges -     a list, containing the point pairs as edges
        time_weights -  a list, the same size as 'edges', containing the corresponding
                    time weights
        time -      a numer, at that time we want to convert the filter network.
                    Default is 1.
    output:
        M -         n*n np.array, adjacent matrix of network.
    """

    # initialization
    n = len(nodes)      # nodes number
    num_edge = len(edges)   # edges number
    M = np.zeros((n, n))    # initialize adjacent matrix

    # find the edges with 'weight' less than or equal to 'time'
    for i in range(num_edge):
        if time_weights[i] <= time:
            M[edges[i][0], edges[i][1]] = 1
            M[edges[i][1], edges[i][0]] = 1
    
    return M

def network_test(data, method='nen', knn=3, time=1.5, metric='euclidean', weighted=False, 
    solid=False, connected=False, verbose=0):
    """
    Construct a network to approximate the manifold in which the data resides.
    Input:
        data -          n*m np.array
        method -        string, 'nen' - non-uniform epsilon neighborhood(default).
                        'fen' - fixed epsilon neighborhood. 
                        'knn' - k nearest neighbors.

    """
    knn = knn
    time = time
    
    n = data.shape[0]
    
    # distance matrix
    matrix_dist = squareform(pdist(data, metric='euclidean'))

    # calculate growth rates matrix
    matrix_nn = np.sort(matrix_dist)    # nearest neighbors matrix
    if knn==0:
        growth_rates = np.ones((1, n))
    else:
        growth_rates = np.mean(matrix_nn[:, range(1, knn+1)], axis=-1)
        growth_rates = np.reshape(growth_rates, (1,n))

    # calculate the time at which point pairs connect
    matrix_grow = growth_rates + np.transpose(growth_rates)
    times = matrix_dist / matrix_grow

    # adjacency matrix
#     if weighted:
#         adm = np.zeros((n,n))
#         for i in range(n):
#             for j in range(i+1,n):
#                 if times[i,j] < time:
#                     adm[i,j] = times[i,j]
#                     adm[j,i] = times[i,j]
#     else:
#         adm = np.zeros((n,n))
#         for i in range(n):
#             for j in range(i+1,n):
#                 if times[i,j] < time:
#                     adm[i,j] = 1
#                     adm[j,i] = 1

    # generate unweighted adjacency matrix
    adm = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            if times[i,j] < time:
                adm[i,j] = 1
                adm[j,i] = 1
    
    # 如果边构成了三角形，就保留，否则就删除
    if solid:
        print(' - removing weak edges ... ', end='', flush=True)
        newadm = np.zeros((n,n))
        for i in range(n-2):
            for j in range(i+1,n-1):
                for k in range(j+1,n):
                    if adm[i,j]*adm[i,k]*adm[j,k]==1:
                        newadm[i,j] = 1
                        newadm[j,i] = 1
                        newadm[i,k] = 1
                        newadm[k,i] = 1
                        newadm[j,k] = 1
                        newadm[k,j] = 1
        adm = newadm
        print('done')
    
    # 保持整个网络连通
    if connected:
        bins = connected_components(adm)
        if len(bins)>1:
            print(' - network have', len(bins), 'components')
        else:
            print('network is connected')

        # 错误：这里有错，每次循环，bins的个数应该减一
        while len(bins)>1:
            print(' - connecting', len(bins), 'compoentes', flush=True)
    #         print(bins)

    #         print(bins[0])
            # 另外一个桶，存放bins[0]之外的点
            others = [i for i in range(n) if i not in bins[0]]
    #         print(others)

            # 找两个桶之间距离最近的边edge1, 并将其连接
            edge1 = [bins[0][0], others[0]]
    #         print('length of edge1 = ', matrix_dist[edge1[0], edge1[1]])
            for i in bins[0]:
                for j in others:
    #                 print(i,j)
                    if matrix_dist[i,j] < matrix_dist[edge1[0],edge1[1]]:
                        edge1 = [i,j]
    #                     print('length of edge1 = ', matrix_dist[edge1[0], edge1[1]])
            adm[edge1[0], edge1[1]] = 1
            adm[edge1[1], edge1[0]] = 1
            print(' - generating the shortest edge', edge1)

            # 再在【bins[0], edge1[1]】和【edge1[0], others】中找一个距离最近的边edge2，并将其连接
            edge2 = []
            for i in bins[0]:
    #             print(i)
    #             print(adm[i,edge1[0]])
    #             print(matrix_dist[i, edge1[1]])
                if adm[i,edge1[0]]==1:
                    if edge2==[]:
                        edge2 = [i, edge1[1]]
    #                     print(edge2, 'initial length = ', matrix_dist[edge2[0], edge2[1]])
                    elif matrix_dist[i, edge1[1]] < matrix_dist[edge2[0],edge2[1]]:
                        edges2 = [i, edge1[1]]
    #                     print(edge2, 'length = ', matrix_dist[edge2[0], edge2[1]])    
    #         print('---')
            for j in others:
    #             print(j)
    #             print(adm[ edge1[1],j ])
    #             print(matrix_dist[ edge1[0],j ] )
                if adm[ edge1[1],j ] == 1 and (matrix_dist[ edge1[0],j ] < matrix_dist[ edge2[0],edge2[1] ]):
                    edge2 = [edge1[0],j]
    #                 print(edge2, 'length = ', matrix_dist[edge2[0], edge2[1]])
            adm[edge2[0], edge2[1]] = 1
            adm[edge2[1], edge2[0]] = 1
            print(' - generating the 2nd shortest edge', edge2)

            # 再次计算连通性
            bins = connected_components(adm)
    #         print(bins)
        
                
    if weighted:
        wadm = np.zeros((n,n))
        for i in range(n):
            for j in range(i+1,n):
                if adm[i,j]==1:
                    wadm[i,j] = times[i,j]
                    wadm[j,i] = times[i,j]
        return wadm
    else:
        return adm