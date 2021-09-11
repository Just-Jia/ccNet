import igraph as ig
import networkx as nx
import leidenalg as la

def adm2graph(net, verbose=0):
    """net -> networkx -> python-igraph"""
    G_nx = nx.from_numpy_array(net)
    g = ig.Graph.from_networkx(G_nx)
    return g

def leiden(g, verbose=0):
    """using leiden algorithm to find a partition, and return labels"""
    par = la.find_partition(g, la.ModularityVertexPartition)
    labels = [-1 for i in range(len(g.degree()))]
    for i in range(len(par)):
        for j in range(len(par[i])):
            labels[par[i][j]] = i
    return labels
    
    