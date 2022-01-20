import numpy as np
import leidenalg

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score


def sigmoid(adm, data, k=4, metric='euclidean', c=0.55, h=1.5, s=30):
    """
    Weighting the edges by Sigmoid function.
    
    Parameters
    ----------
    
    adm: 2d np.array
        adjacency matrix of network
    data:
        data 
    k: int
        number of neighbors for reference
    c: float
        center of sigmoid function
    h: float
        height of sigmoid function
    s: folat
        slope of sigmoid function
        
    Returns
    -------
    
    wadm: 2d np.array
        weighted adjacency matrix
    """

    n = data.shape[0]

    # distance matrix
    matrix_dist = squareform(pdist(data, metric=metric))
    
    # calculate growth rates matrix
    if k==0:
        growth_rates = np.ones((1, n))
    else:
        if metric == 'cosine':
            matrix_nn = np.sort(matrix_dist)    # nearest neighbors matrix
            growth_rates = np.mean(matrix_nn[:, range(1, k+1)], axis=-1)
        elif metric == 'euclidean':
            nbrs = NearestNeighbors(n_neighbors=k+1).fit(data)
            distances, indices = nbrs.kneighbors(data)
            growth_rates = np.mean(distances[:,1:], axis=-1)
        else:
            print('please use euclidean of cosine as metirc! ')
        growth_rates = np.reshape(growth_rates, (1,n))
        
    # calculate the time at which point pairs connect
    matrix_grow = growth_rates + np.transpose(growth_rates)
    times = matrix_dist / matrix_grow
        
    wadm = h / (1 + np.exp(s*(times - c)))
    wadm = wadm * adm
    
    return wadm

def get_igraph_from_adjacency(adjacency, directed=None):
    """Get igraph graph from adjacency matrix."""
    import igraph as ig

    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])  # this adds adjacency.shape[0] vertices
    g.add_edges(list(zip(sources, targets)))
    try:
        g.es['weight'] = weights
    except KeyError:
        pass
    if g.vcount() != adjacency.shape[0]:
        logg.warning(
            f'The constructed graph has only {g.vcount()} nodes. '
            'Your adjacency matrix contained redundant nodes.'
        )
    return g

def leiden_cluster(wadm, resolution=0.1):
    # 将邻接矩阵转变为igraph
    g = get_igraph_from_adjacency(wadm, directed=True)


    # 设置相关参数
    partition_type = leidenalg.RBConfigurationVertexPartition

    partition_kwargs={}
    partition_kwargs['weights'] = np.array(g.es['weight']).astype(np.float64)
    partition_kwargs['n_iterations'] = -1
    partition_kwargs['seed'] = 0
    partition_kwargs['resolution_parameter'] = resolution

    par = leidenalg.find_partition(g, partition_type, **partition_kwargs)

    # 划分转化为标签
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


def compute_score(predict_labels, labels, verbose=0):
    """计算聚类得分"""
    ari = adjusted_rand_score(predict_labels, labels)
    ami = adjusted_mutual_info_score(predict_labels, labels)
    nmi = normalized_mutual_info_score(predict_labels, labels)
    
    if verbose==1:
        print('-'*30)
        print('ari\tami\tnmi')
        print('{:.4f}  {:.4f}  {:.4f}'.format(ari, ami, nmi))
        
    return [ari, ami, nmi]