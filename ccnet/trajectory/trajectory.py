import numpy as np
import networkx as nx

def pseudotime(A, source=0, verbose=0):
    """calculate pseudotime"""
    
    G = nx.from_numpy_matrix(A)
    length = nx.single_source_dijkstra_path_length(G, source)
    t = [length[i] for i in range(len(length))]
    
    return t