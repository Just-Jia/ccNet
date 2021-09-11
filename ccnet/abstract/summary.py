"""
将可视化，聚类，以及轨迹推断进行总结，得到一个有向图
"""
from graphviz import Digraph
import numpy as np

def summary(adm, predict_labels, predict_time, view=True, verbose=1):
    """
    Compute the summary graph, which containing the clustering and trajectory inference result.
    """
# 计算每个聚类的平均发育时间

    print(' - checking the parameter ... ', end='')
    
    if adm.shape[0] != adm.shape[1]:
        raise Exception('adm must be a square')
    if adm.shape[0] != len(predict_labels) or adm.shape[0] != len(predict_time) or len(predict_labels) != len(predict_time):
        raise Exception('order of adm, lenght of pos and predict_labels must be equal')
        
    print('done')
    
    
    # 聚类标签列表
    clusters = sorted( set(predict_labels) )
    ncbins = len(clusters)

    print(' - putting the cells into bins ... ', end='')
    # 桶里面放各个cluster的节点
    # 初始化桶
    cbins = {}
    for i in range(ncbins):
        cbins[ clusters[i] ] = []
    
    for i in range( len(predict_labels) ):
        cbins[ predict_labels[i] ].append( i )
    print('done')
    
    print(' - computing the mean develop time for each bin/cluster ... ', end='')
    # 计算每个聚类的平均发育时间
    # 初始化
    ctime = {}
    for i in range(ncbins):
        ctime[ clusters[i] ] = 0.0

    for i in range(ncbins):
        num = len(cbins[clusters[i]])
#         print(i, num)
        for j in range(num):
            ctime[clusters[i]] += predict_time[cbins[clusters[i]][j]]
        ctime[clusters[i]] /= num
    ## 注意：这里先加后除，有可能会溢出
    print('done')

    print(' - ctime = ', ctime)

    
    print(' - computing the connectivities between bins/clusters ... ', end='')   
    # 计算邻接矩阵
    adm2 = np.zeros((ncbins,ncbins))
    for i in range(ncbins):
        for j in range(ncbins):
            # 第i个桶中的结点为cbins[clusters[i]], 第j个桶中的结点为cbins[clusters[j]]，
            for r in cbins[clusters[i]]:
                for c in cbins[clusters[j]]:
                    if adm[r,c]>0:
                        adm2[i,j] += 1
    print('done')
    
    print(' - computing the development graph ... ', end='')
    # 计算聚类之间的发育图development graph
    g = Digraph('summary', filename='abstraction.gv',
                node_attr={'color': 'lightblue2', 'style': 'filled'})
                       
    g.attr(size='6,6')
    
    # 计算每个聚类的发育时间
    for i in range(ncbins):
        num_edge = int(adm2[i,i]/2) # total number of edges within cluster i
        num_cell = len(cbins[i]) # number of cells in cbins[i]
        mean_time = ctime[i] # mean develop time in cluster i
        
        # 增加节点，及其相关信息
        g.node(str(i), label='Cluster {:}\ncells = {:}\nedges = {:}\nday = {:.2}'.format(i, num_cell, num_edge, mean_time))
        
    # 计算聚类之间的边
    for i in range(ncbins):
        for j in range(i+1,ncbins):
            if adm2[i,j]>0:
                if ctime[i] > ctime[j]:
                    g.edge(str(j), str(i), label='edge = {:}'.format(int(adm2[i,j])))
                else:
                    g.edge(str(i), str(j), label='edge = {:}'.format(int(adm2[i,j])))
    print('done')
    if view is True:
        g.view()
    return g