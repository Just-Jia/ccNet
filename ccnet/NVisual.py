"""
NVisual, which is the abb. of Network Vialization
TODO: 检查network_visualize 中的pos 是按值传递还是引用传递。也就是，如果函数中对pos有修改，
    函数外面的pos是否也会改变。
"""

import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

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
