import numpy as np

def pseudotime(A, source=0, verbose=0):
    """
    Perform trajectory inference.
    """
    return Dijkstra(A, source=source, verbose=verbose)

def bifurcation(A, d=1.3, verbose=0):
    """
    Calculate the bifurcation point.
    """
    return branches(A, d=d, verbose=verbose)

# 计算最短路径
def Dijkstra(A, source=0, verbose=0):
    """
    用来计算最短路径问题。这个过程就像相连的气球沉到水里
    input：
        A - 2*2 np.array, 表示邻接矩阵
        source - number, 表示源节点
        verbose - number, 是否显示调试信息，1 - 显示，0 - 不显示，默认为不显示
    output：
        distances - list, 表示每个节点到源节点的最短距离
    
    """
    
    verbose = verbose
    n = A.shape[0]
    visited = [False for i in range(n)]
    distances = [0 for i in range(n)]
    
    visited[source] = True
    distances[source] = 0

    deleted_edge = 0
    deleted_node = source

    # step 1
    edges = {}
    for j in range(n):
        if A[deleted_node, j] > 0 and visited[j] is False:
            edges[(deleted_node,j)] = distances[deleted_node] + A[deleted_node, j]
    
    if verbose == 1:
        print(' - start:')
        for edge, value in edges.items(): print('   ', edge, value)

    # 如果字典不为空，就一直循环
    while len(edges) > 0:
        # 找最小的值，以及对应的边
        minvalue = float('inf')
        for key, value in edges.items():
            if value < minvalue:
                deleted_edge = key
                minvalue = value
        deleted_node = deleted_edge[1]

        # 从字典中删掉最小值对应的边, 以及与第二个节点相连的边，同时更新visted和distances
        del edges[deleted_edge]
        will_delete = []
        for edge in edges.keys():
            if edge[1] == deleted_node: will_delete.append(edge)
        for i in range(len(will_delete)): del edges[will_delete[i]]
        visited[deleted_node] = True
        distances[deleted_node] = minvalue

        if verbose == 1:
            print(' - after delete edges:')
            for edge, value in edges.items(): print('   ', edge, value)

        # 找新边
        for j in range(n):
            if A[deleted_node, j] > 0 and visited[j] is False:
                edges[(deleted_node,j)] = distances[deleted_node] + A[deleted_node, j]
                
        if verbose == 1:
            print(' - after add new edges:')
            for edge, value in edges.items(): print('   ', edge, value)
                
    return distances

# dist = Dijkstra(A, 0, verbose=1)

def branches(A, d=1.3, verbose=0):
    """
    Compute the branch points.
    Input:
        A   - n*n np.array(), adjacency matrix
        d -   number, distance in fdl-neighbor
        verbose - number, verbosity
    Output:
        nodes - dictionary, key is the number of branches, value is a list 
                containg the nodes with the branches equal key
    """
    
    if A.shape[0]!=A.shape[1]:
        print('ERROR: A is not a square!')
        
    n = A.shape[0]
    nodes=[]
    for i in range(n):
        # 计算每个节点的fdl_neighbors，它们构成一个邻居网，更加这个网的连通分支数来确定点的分支类型
        neighbors = fdl_neighbors(A, source=i, d=d)
        nei_net = A.take(neighbors, axis=0).take(neighbors, axis=1)
        cc = connected_components(nei_net)
        if len(cc)>2: 
            nodes.append(i)
#         if i%10==0:
        print('\r', i , '/', n, end='', flush=True)
    return nodes

def branches_Floyd_Warshall(A, d=1.3, maxd=2):
    """
    计算分支点，采用Floy-Warshall 来计算最短路径距离
    """
    distm = A.copy()

    float("inf")
    n = A.shape[0]

    # 初始化距离矩阵
    print(' - now initializing the distance matrix')
    for i in range(n):
        for j in range(i+1,n):
            if distm[i,j]==0:
                distm[i,j]=float("inf")
                distm[j,i]=float("inf")
    
    print(' - computing')
    for k in range(n):
        print('\r', k, '/', n, end='', flush=True)
        for i in range(n):
#             print('i=', i, end='')
            for j in range(i+1,n):
                if distm[i,j] > distm[i,k] + distm[k,j]:
                    distm[i,j] = distm[i,k] + distm[k,j]
                    distm[j,i] = distm[i,j]
    
    print()
    nodes = []
    for i in range(n):
        print('\r', i, '/', n, end='', flush=True)
        neighbors = [j for j in range(n) if distm[i,j]>d and distm[i,j]<maxd]
        nei_net = A.take(neighbors, axis=0).take(neighbors, axis=1)
        cc = connected_components(nei_net)
        if len(cc)>2: 
            nodes.append(i)
            
    print()
    return nodes


def fdl_neighbors(A, source=0, d=1.5, verbose=0):
    """
    Compute the first d-large neighbors.
    If node j is the fdl-neighbors of node i, then the shorest path lenght from i to j is 
    larger then d, but the length of passing path node is less than d.
    
    Input:
        A -        n*n np.array(), represent adjacency matrix, it can be weighted
        source -   number
        d -        number
        verbose -  verbosity.
    Output
       neighbors - a list, containing the fdl-neighbors
    """
    
#     verbose = 1
    
    n = A.shape[0]
    visited = [False for i in range(n)]
    distances = [0 for i in range(n)]

    # 更新
    deleted_node = source
    visited[source] = True
    distances[source] = 0

    # edges, of the form [node1, node2, distance_to_source, is_beyond_d]
    edges = [[],[],[],[]]
    for j in range(n):
        if A[source, j]>0 and not visited[j]: 
            edges[0].append(source)
            edges[1].append(j)
            edges[2].append(A[source, j])
            edges[3].append(A[source, j]>d)
    if verbose == 1:
        for i in range(len(edges[0])):
            print(edges[0][i],edges[1][i],edges[2][i],edges[3][i])

    edges_lessd = [i for i in range(len(edges[3])) if edges[3][i] == False]

    while len(edges[0])>0 and len(edges_lessd):
        # 找最小值，以及对应的边, 这里要保证第四列小于d
        node_indices = [i for i in range(len(edges[0])) if edges[3][i] == False]
        node_distances = [edges[2][node_indices[i]] for i in range(len(node_indices))]
        min_distance = min(node_distances)
        ind = edges[2].index(min_distance)

        deleted_node = edges[1][ind]

        # 从字典中删掉最小值对应的边, 以及与第二个节点相连的边，同时更新visted和distances
        rows = [i for i in range(len(edges[1])) if edges[1][i]==deleted_node]
        for i in range(len(rows)-1, -1, -1):
            for j in range(4):
                edges[j].pop(rows[i])

        visited[deleted_node] = True
        distances[deleted_node] = min_distance

        if verbose == 1:
            for i in range(len(edges[0])):
                print(edges[0][i],edges[1][i],edges[2][i],edges[3][i])

        # 找新边
        for j in range(n):
            if A[deleted_node, j]>0 and not visited[j]: 
                edges[0].append(deleted_node)
                edges[1].append(j)
                dist2j = distances[deleted_node] + A[deleted_node, j]
                edges[2].append(dist2j)
                edges[3].append(dist2j>d)

        if verbose == 1:
            for i in range(len(edges[0])):
                print(edges[0][i],edges[1][i],edges[2][i],edges[3][i])
        
        edges_lessd = [i for i in range(len(edges[3])) if edges[3][i] == False]
    
    # delete repeated nodes
    neighbors = list(set(edges[1]))
    return neighbors


def connected_components(A, verbose=0):
    """
    Compute the connected components of network A
    Input:
        A  - n*n np.array(), adjacency matrix of the network
    Output:
        bins - a dictionary, key is number represent id of bin,
                and value is list containing each connected components
    """
    if A.shape[0] != A.shape[1]:
        print('ERROR: A is not square')
    
    n = A.shape[0]
    
    visited = [0 for i in range(n)] # 0 - unvisited; 1 - visited
    bins = {}

    num = -1 # number of bin, or id of bin
    while sum(visited) < n:
        # initialize the bin
        num = num + 1
        bins[num] = []

        # get an unvisited node
        node = 0
        while visited[node]==1: node = node + 1

        # put this unvisited node into current bin, and mark it as visited
        bins[num].append(node)
        visited[node] = 1
        

        # find the visiting edges, namely, the first node is visited 
        # and second one is unvisited
        edges = [(node, j) for j in range(n) if A[node,j]>0 and visited[j]==0]

        while len(edges)>0:
            # find the node we want to delete
            deleted_node = edges[0][1]

            # find the rows we want to delete, and delete them
            rows = [i for i in range(len(edges)) if edges[i][1]==deleted_node]
            for i in range(len(rows)-1, -1, -1):
                edges.pop(rows[i])

            # put the deleted_node into current bin, and mark it as visited
            bins[num].append(deleted_node)
            visited[deleted_node] = 1

            # add new edges
            for j in range(n):
                if A[deleted_node, j]>0 and visited[j]==0:
                    edges.append((deleted_node, j))
    return bins

