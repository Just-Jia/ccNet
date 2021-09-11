import numpy as np
import community as community_louvain
import networkx as nx

def louvain(adm, n_iters=1):
    """Community detection by louvain"""
    
    # construct netwotkx object from adjacent matrix
    G = nx.from_numpy_array(adm)
    partition = community_louvain.best_partition(G)
    m = community_louvain.modularity(partition, G)
    
    i = 1
    print()
    while i < n_iters:
        partition0 = community_louvain.best_partition(G)
        m0 = community_louvain.modularity(partition0, G)
        print('\r     {} of {}, modularity = {:.4f}'.format(i, n_iters, m), sep='', end='')
        i += 1
        if m0 > m:
            partition = partition0
            m = m0
    # partition 是一个字典，根据键来排序
    labels = [partition[node] for node in sorted(partition.keys())]
    return labels