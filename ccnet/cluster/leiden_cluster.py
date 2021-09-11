import leidenalg
import numpy as np

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

def leiden_cluster(wadm):
    # 将邻接矩阵转变为igraph
    g = get_igraph_from_adjacency(wadm, directed=True)


    # 设置相关参数
    partition_type = leidenalg.RBConfigurationVertexPartition

    partition_kwargs={}
    partition_kwargs['weights'] = np.array(g.es['weight']).astype(np.float64)
    partition_kwargs['n_iterations'] = -1
    partition_kwargs['seed'] = 0
    partition_kwargs['resolution_parameter'] = 1

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