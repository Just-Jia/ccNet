"""
NVisual, which is the abb. of Network Vialization
TODO: 检查network_visualize 中的pos 是按值传递还是引用传递。也就是，如果函数中对pos有修改，
    函数外面的pos是否也会改变。
"""

import numpy as np
import networkx as nx
from .fa2 import ForceAtlas2

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def plot_net(ax, pos, adm=None, edges=True,
             cmap=plt.cm.rainbow,
             node_color='tab:blue',
             node_size=10,  
             edge_color='black', 
             edge_width=0.2,
             verbose=0):
    """
    Plot the network using the given layout.
    Input:
        ax - 
        pos -           dictionary, containing the position of nodes.
        edges -         logical value, True - plot the edges(default)
                            False - does not plot the edges
        verbose -   number, verbosity. 0 or 1.
    Output:
    """
    
    # number of nodes
    n = len(pos)
    
    if edges is True:
        if adm is None:
            raise ValueError('Parameter \'adm\' is required')
        if n != adm.shape[0]:
            raise ValueError('length of pos must equal node number')
            
        edgelist = [(i,j) for i in range(n-1) for j in range(i+1, n) if adm[i,j]>0]
        edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])

        edge_collection = LineCollection(edge_pos, 
                                         colors=edge_color,
                                        linewidths=edge_width)
        edge_collection.set_zorder(1)  # edges go behind nodes
        ax.add_collection(edge_collection)
        
    xy = np.asarray([pos[v] for v in range(n)])
    out = ax.scatter(xy[:,0], 
                     xy[:,1], 
                     s=node_size, 
                     c=node_color,
                     cmap=cmap)

    return out

def visualize(M, pos={}, node_color='tab:blue', edge_width=0.1):
    """
    Visualize the network given by adjacency matrix.
    """
    return network_visualize(M, pos=pos, node_color=node_color, edge_width=edge_width)

def network_visualize(M, pos={}, node_color='tab:blue', edge_width=0.1):
    """
    Network visualization module, which depend on the library 'networkx'. 
    input:
        M -             n*n np.array, represents adjacent matrix.
        node_color -    a color string or value list, used to draw color of node.
                        This parameter is similar to parameter 'c' in function matplotlib.pyplot.
                        So refer to 'c' for details.
        edge_width -    a number, specifies the width of edges.
    output:
        pos -           dictionary, containing the position of nodes.
    """
    n=M.shape[0]

    # construct network, and save it to .gexf file
    # construct nx class from adjacent matrix
    G = nx.from_numpy_array(M)
    if pos=={}:
        print(' - calculating network layout ... ', end='')
        # calculate layout
        pos = nx.nx_agraph.graphviz_layout(G, 'sfdp', '-Goverlap=false -GK=0.1')
        print('done')
    else:
        print(' - using the network layout passed ... ')
    # color value
    # colors = [v for v in range(n)]
    node_color = node_color     # 'tab:blue'
    edge_width = edge_width     # 0.1
    # fig, ax = plt.subplots()
    plt.figure(figsize=(11, 8))
    nx.draw_networkx(G, 
                     pos=pos, # position of nodes
                     node_color=node_color, # colors of node
                     cmap=plt.cm.rainbow, # color map
                     with_labels=False, # if draw label for each node
                     node_size=20, # size of node
                     linewidths=None, # Line width of symbol border
                     width=edge_width, # Line width of edges
                     edge_color='black', # color of edge. 'grey'
                     alpha=0.9, # The node and edge transparency
                    )
    plt.show()

    return pos
