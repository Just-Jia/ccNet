import networkx as nx
from .fa2 import ForceAtlas2

def layout(M, method='spring-electrical', size=None, init_pos=None, seed=4, niter1=3000, niter2=200, 
    outbound1=False, outbound2=False,
    linlog1=False, linlog2=False, 
    gravity1=1.0, gravity2=1.0,
    verbose=0):
    """
    Compute the layout of the network.
    Input:
        M -             n*n np.array, represents adjacent matrix.
        method -    string, 'spring-electrical', spring-electrical model algorithm.
                        'force-atlas-2'
        verbose -   number, verbosity. 0 or 1
    Output:
        pos -           dictionary, containing the position of nodes.
    """
    print(' - calculating network layout ... ', end='')
    
    if method in ['spring-electrical', 'spring_electrical', 'SE', 'se']:
        pos = spring_electrical(M)
    elif method in ['ForceAtlas2', 'fa2', 'force_atlas_2']:
        pos = force_atlas_2(M, seed=seed, init_pos=init_pos, size=size, niter1=niter1, niter2=niter2, 
            outbound1=outbound1, outbound2=outbound2,
            linlog1=linlog1, linlog2=linlog2,
            gravity1=gravity1, gravity2=gravity2)
    
    print('done')
    return pos

def spring_electrical(M):
    """
    使用电子弹簧模型生成网络的布局
    """
    # construct nx class from adjacency matrix, and calculate layout
    G = nx.from_numpy_array( M )
    pos = nx.nx_agraph.graphviz_layout(G, 'sfdp', '-Goverlap=false -GK=0.1')
    return pos

def force_atlas_2(M, seed=4, init_pos=None, size=None, niter1=3000, niter2=200, outbound1=False, outbound2=False, 
    linlog1=False, linlog2=False,
    gravity1=1.0, gravity2=1.0):
    # 计算布局
    forceatlas2 = ForceAtlas2(
                        graph=M,
                        iterations=niter1,
                        pos=init_pos,
    
                        # Behavior alternatives
                        outboundAttractionDistribution=outbound1,  # Dissuade hubs
                        linLogMode=linlog1,  # 
                        adjustSizes=False,  # Prevent overlap
                        sizes=10,
                        edgeWeightInfluence=1.0,

                        # Performance
                        jitterTolerance=1.0,  # Tolerance
                        barnesHutOptimize=True, # True, by default
                        barnesHutTheta=1.2,
                        multiThreaded=False,  # NOT IMPLEMENTED

                        # Tuning
                        scalingRatio=2.0,
                        strongGravityMode=False,
                        gravity=gravity1,
                        
                        # random
                        seed=seed,

                        # Log
                        verbose=True)
    positions = forceatlas2.forceatlas2()
    if (size is None) or size==0:
        return positions
    else:
        forceatlas22 = ForceAtlas2(
                        graph=M,
                        iterations=niter2,
                        pos=positions,
    
                        # Behavior alternatives
                        outboundAttractionDistribution=outbound2,  # Dissuade hubs
                        linLogMode=linlog2,  # 
                        adjustSizes=True,  # Prevent overlap
                        sizes=size,
                        edgeWeightInfluence=1.0,

                        # Performance
                        jitterTolerance=1.0,  # Tolerance
                        barnesHutOptimize=False, # True, by default
                        barnesHutTheta=1.2,
                        multiThreaded=False,  # NOT IMPLEMENTED

                        # Tuning
                        scalingRatio=2.0,
                        strongGravityMode=False,
                        gravity=gravity2,
                        
                        # random
                        seed=4,

                        # Log
                        verbose=True)
        positions2 = forceatlas22.forceatlas2()
        return positions2


def force_atlas_2_old(M):
    """
    使用ForceAtlas2进行网络布局
    """
    # construct nx class from adjacency matrix, and calculate layout
    G = nx.from_numpy_array( M )
    forceatlas2 = ForceAtlas2(
                            # Behavior alternatives
                            outboundAttractionDistribution=True,  # Dissuade hubs
                            linLogMode=False,  # NOT IMPLEMENTED
                            adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                            edgeWeightInfluence=1.0,

                            # Performance
                            jitterTolerance=1.0,  # Tolerance
                            barnesHutOptimize=True,
                            barnesHutTheta=1.2,
                            multiThreaded=False,  # NOT IMPLEMENTED

                            # Tuning
                            scalingRatio=2.0,
                            strongGravityMode=False,
                            gravity=1.0,

                            # Log
                            verbose=True)

    pos = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=2000)
    return pos