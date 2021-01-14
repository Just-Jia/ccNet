"""
TODO: 增加其他的社团检测算法:比如谱聚类，层次聚类，leiden
[1]From Louvain to Leiden: guaranteeing well-connected communities
[2] Pons, P. & Latapy, M. Computing communities in large networks using random walks. Computer
and Information Sciences - ISCIS 284 (2005).
[3]S. Fortunato. Community detection in graphs. Physics Reports, 486(3-5):75{174,
2010.
"""
import community as community_louvain
import networkx as nx
import numpy as np

import traceback

def clustering(adm, method='louvain'):
    """
    Operate clustering analysis.
    """
    return community_detect(adm, method=method)

def community_detect(adm, method='louvain'):
    """
    Perform community detection for complex network.
    input:
        adm -       n*n np.array, represent adjacent matrix of network.
        method -    string, the community detection algorithm used here.
                    Default 'louvain'.
    output:
        labels -    list, containing cluster labels coresponding to each point.
                    It's lenght equals the node number, namely the order of 
                    'adm'.
    """
    print(' - detecting communities ... ', end='')

    # construct netwotkx object from adjacent matrix
    G = nx.from_numpy_array(adm)
    if method=='louvain':
        partition = community_louvain.best_partition(G)
        # partition 是一个字典，根据键来排序
        labels = [partition[node] for node in sorted(partition.keys())]
    else:
        raise Exception("Wrong parameter: method must be one of louvain!")

    print('done')
    return labels