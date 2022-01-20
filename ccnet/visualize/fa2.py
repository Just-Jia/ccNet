# This is the fastest python implementation of the ForceAtlas2 plugin from Gephi
# intended to be used with networkx, but is in theory independent of
# it since it only relies on the adjacency matrix.  This
# implementation is based directly on the Gephi plugin:
#
# https://github.com/gephi/gephi/blob/master/modules/LayoutPlugin/src/main/java/org/gephi/layout/plugin/forceAtlas2/ForceAtlas2.java
#
# For simplicity and for keeping code in sync with upstream, I have
# reused as many of the variable/function names as possible, even when
# they are in a more java-like style (e.g. camelcase)
#
# I wrote this because I wanted an almost feature complete and fast implementation
# of ForceAtlas2 algorithm in python
#
# NOTES: Currently, this only works for weighted undirected graphs.
#
# Copyright (C) 2017 Bhargav Chippada <bhargavchippada19@gmail.com>
#
# Available under the GPLv3

import random
import time

import numpy
import scipy
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# from . 
from . import tool as fa2util

class ForceAtlas2:
    def __init__(self,
                graph, 
                iterations,
                pos=None,
                 # Behavior alternatives
                 outboundAttractionDistribution=False,  # Dissuade hubs
                 linLogMode=False,  # NOT IMPLEMENTED
                 adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                 sizes=0,
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

                 # random
                 seed=None,

                 # Log
                 verbose=True):
        self.graph = graph
        self.iterations = iterations
        self.pos = pos
        # n = len(graph)
        n = graph.shape[0]
        if isinstance(sizes, int) or isinstance(sizes, float):
            self.sizes = [sizes for i in range(n)]
        else:
            self.sizes = sizes

        # assert linLogMode == multiThreaded == False, "You selected a feature that has not been implemented yet..."
        self.outboundAttractionDistribution = outboundAttractionDistribution
        self.linLogMode = linLogMode
        self.adjustSizes = adjustSizes
        # self.sizes = sizes
        self.edgeWeightInfluence = edgeWeightInfluence
        self.jitterTolerance = jitterTolerance
        self.barnesHutOptimize = barnesHutOptimize
        self.barnesHutTheta = barnesHutTheta
        self.scalingRatio = scalingRatio
        self.strongGravityMode = strongGravityMode
        self.gravity = gravity
        self.seed = seed
        self.verbose = verbose

        self.nodes, self.edges = self.init()

    # def init(self,
    #          G,  # a graph in 2D numpy ndarray format (or) scipy sparse matrix format
    #          pos=None  # Array of initial positions
    #          ):
    def init(self):
        G = self.graph
        pos = self.pos

        isSparse = False
        if isinstance(G, numpy.ndarray):
            # Check our assumptions
            assert G.shape == (G.shape[0], G.shape[0]), "G is not 2D square"
            assert numpy.all(G.T == G), "G is not symmetric.  Currently only undirected graphs are supported"
            assert isinstance(pos, numpy.ndarray) or (pos is None), "Invalid node positions"
        elif scipy.sparse.issparse(G):
            # Check our assumptions
            assert G.shape == (G.shape[0], G.shape[0]), "G is not 2D square"
            assert isinstance(pos, numpy.ndarray) or (pos is None), "Invalid node positions"
            G = G.tolil()
            isSparse = True
        else:
            assert False, "G is not numpy ndarray or scipy sparse matrix"

        #Set random seed
        if not self.seed is None:
            random.seed(self.seed)

        # Put nodes into a data structure we can understand
        nodes = []
        for i in range(0, G.shape[0]):
            n = fa2util.Node()
            if isSparse:
                n.mass = 1 + len(G.rows[i])
            else:
                n.mass = 1 + numpy.count_nonzero(G[i])
            n.size = self.sizes[i]
            n.old_dx = 0
            n.old_dy = 0
            n.dx = 0
            n.dy = 0
            if pos is None:
                n.x = random.random()
                n.y = random.random()
            else:
                n.x = pos[i][0]
                n.y = pos[i][1]
            nodes.append(n)

        # Put edges into a data structure we can understand
        edges = []
        es = numpy.asarray(G.nonzero()).T
        for e in es:  # Iterate through edges
            if e[1] <= e[0]: continue  # Avoid duplicate edges
            edge = fa2util.Edge()
            edge.node1 = e[0]  # The index of the first node in `nodes`
            edge.node2 = e[1]  # The index of the second node in `nodes`
            edge.weight = G[tuple(e)]
            edges.append(edge)

        return nodes, edges

    # Given an adjacency matrix, this function computes the node positions
    # according to the ForceAtlas2 layout algorithm.  It takes the same
    # arguments that one would give to the ForceAtlas2 algorithm in Gephi.
    # Not all of them are implemented.  See below for a description of
    # each parameter and whether or not it has been implemented.
    #
    # This function will return a list of X-Y coordinate tuples, ordered
    # in the same way as the rows/columns in the input matrix.
    #
    # The only reason you would want to run this directly is if you don't
    # use networkx.  In this case, you'll likely need to convert the
    # output to a more usable format.  If you do use networkx, use the
    # "forceatlas2_networkx_layout" function below.
    #
    # Currently, only undirected graphs are supported so the adjacency matrix
    # should be symmetric.
    # def forceatlas2(self,
    #                 G,  # a graph in 2D numpy ndarray format (or) scipy sparse matrix format
    #                 pos=None,  # Array of initial positions
    #                 iterations=100  # Number of times to iterate the main loop
    #                 ):
    def forceatlas2(self):
        # Initializing, initAlgo()
        # ================================================================

        # speed and speedEfficiency describe a scaling factor of dx and dy
        # before x and y are adjusted.  These are modified as the
        # algorithm runs to help ensure convergence.
        iterations = self.iterations

        speed = 1.0
        speedEfficiency = 1.0
        nodes = self.nodes
        edges = self.edges

        outboundAttCompensation = 1.0
        if self.outboundAttractionDistribution:
            outboundAttCompensation = numpy.mean([n.mass for n in nodes])
        # ================================================================

        # Main loop, i.e. goAlgo()
        # ================================================================


        # Each iteration of this loop represents a call to goAlgo().
        niters = range(iterations)
        # if self.verbose:
        #     niters = tqdm(niters)
        for _i in niters:
            print('\r - {} / {}'.format(_i, niters), end='')
            for n in nodes:
                n.old_dx = n.dx
                n.old_dy = n.dy
                n.dx = 0
                n.dy = 0

            # Barnes Hut optimization
            if self.barnesHutOptimize:
                # barneshut_timer.start()
                rootRegion = fa2util.Region(nodes)
                rootRegion.buildSubRegions()
                # barneshut_timer.stop()

            # Charge repulsion forces
            # repulsion_timer.start()
            # parallelization should be implemented here
            if self.barnesHutOptimize:
                rootRegion.applyForceOnNodes(nodes, self.barnesHutTheta, self.scalingRatio, self.adjustSizes)
            else:
                    fa2util.apply_repulsion(nodes, self.scalingRatio, self.adjustSizes)
            # repulsion_timer.stop()

            # Gravitational forces
            # gravity_timer.start()
            fa2util.apply_gravity(nodes, self.gravity, scalingRatio=self.scalingRatio, useStrongGravity=self.strongGravityMode)
            # gravity_timer.stop()

            # If other forms of attraction were implemented they would be selected here.
            # attraction_timer.start()
            fa2util.apply_attraction(nodes, edges, outboundAttCompensation, self.edgeWeightInfluence,\
                self.linLogMode, self.outboundAttractionDistribution, self.adjustSizes)
            # attraction_timer.stop()

            # Adjust speeds and apply forces
            # applyforces_timer.start()
            values = fa2util.adjustSpeedAndApplyForces(nodes, speed, speedEfficiency, self.jitterTolerance, self.adjustSizes)
            speed = values['speed']
            speedEfficiency = values['speedEfficiency']
            # applyforces_timer.stop()

        # if self.verbose:
        #     if self.barnesHutOptimize:
        #         barneshut_timer.display()
        #     repulsion_timer.display()
        #     gravity_timer.display()
        #     attraction_timer.display()
        #     applyforces_timer.display()
        # ================================================================
        # return [(n.x, n.y) for n in nodes]
        return numpy.array([[n.x, n.y] for n in nodes])

    

    # def positions(self):
    #     """计算节点的坐标"""
    #     pos = [[n['x'], n['y']] for n in self.nodes]
    #     return np.array(pos)
    
    def plot(self, ax, pos, nodes=True, ann=True, edges=True, s=5):
        """绘制网络"""
        
        # 画点
        if nodes:
            # get position
            # pos = self.positions()
            ax.scatter(pos[:,0], pos[:,1], s=s)
        
        # 画注释
        if ann:
            for i in range(0, len(pos[:,0])):
                ax.text(pos[i,0],pos[i,1], i)
            
        # 画线
        if edges:
            lines = []
            for e in self.edges:
                sn = self.nodes[e.node1]
                tn = self.nodes[e.node2]
                lines.append(((sn.x, sn.y), (tn.x, tn.y)))
            ln_coll = LineCollection(lines)
            ax.add_collection(ln_coll)

    
