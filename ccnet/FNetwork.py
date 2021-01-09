"""
对数据进行预处理，建立过滤网络和过滤值下的网络
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform



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
