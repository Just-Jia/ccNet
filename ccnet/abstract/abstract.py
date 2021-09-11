import numpy as np

def abstract(adm=None, pos=None, predict_labels=None, ordering=None, cvalue=0, verbose=1):
    """
    Compute the abstraction of clustering (and trajectory inference)
    Input:
        ax - 
        adm -        n*n np.array, adjacency matrix of network derived network.
        pos -        dict, n key-values pairs, position of node, derived visualize.
        predict_labels - list, n elements, containing the clustering labels
        ordering -   list, n elements, containg the prediction labels
        verbose -    number, verbosity, 0 - or 1 - default
    Output:
    """
#     if ax is None:
#         cf = plt.gcf()
#         ax = cf.gca()
    
    if adm.shape[0] != adm.shape[1]:
        raise Exception('adm must be a square')
    if adm.shape[0] != len(pos) or adm.shape[0] != len(predict_labels) or len(pos) != len(predict_labels):
        raise Exception('order of adm, lenght of pos and predict_labels must be equal')
    
# pos
# predict_labels
# adm

# 计算抽象图

    # 聚类标签列表
    clusters = sorted( set(predict_labels) )
    ncbins = len(clusters)

    # 桶里面放各个cluster的节点
    # 初始化桶
    cbins = {}
    for i in range(ncbins):
        cbins[ clusters[i] ] = []

    for i in range( len(predict_labels) ):
        cbins[ predict_labels[i] ].append( i )
    
    # 计算每个桶的位置
    cpos = {}
    for i in range(ncbins):
        cpos[ clusters[i] ] = [0.0, 0.0]

    for i in range(ncbins):
        num = len(cbins[clusters[i]])
#         print(i, num)
        for j in range(num):
    #         print(pos[cbins[clusters[i]][j] ])
            cpos[clusters[i]][0] += pos[cbins[clusters[i]][j] ][0]
            cpos[clusters[i]][1] += pos[cbins[clusters[i]][j] ][1]
        cpos[clusters[i]][0] /= num
        cpos[clusters[i]][1] /= num
    ## 注意：这里先加后除，有可能会溢出
    
    # 计算邻接矩阵, 实际上，它是连边矩阵，i行j列表示它的连边数，对角线是实际连边的二倍
    adm2 = np.zeros((ncbins,ncbins))
    for i in range(ncbins):
        for j in range(ncbins):
            # 第i个桶中的结点为cbins[clusters[i]], 第j个桶中的结点为cbins[clusters[j]]，
            for r in cbins[clusters[i]]:
                for c in cbins[clusters[j]]:
                    if adm[r,c]>0:
                        adm2[i,j] += 1
    # 注意：这里需要修改，当i等于j时，adm2中边算重了
    if cvalue==0:
        return adm2, cpos
    else:
        # 设置阈值cvalue，过滤掉其中的一些边
        adm3 = adm2.copy()
#         cvalue = 2
        for i in range(adm3.shape[0]):
            for j in range(i, adm3.shape[0]):
                if adm3[i,j]<cvalue:
                    adm3[i,j] = 0.0
                    adm3[j,i] = 0.0
        return adm3, cpos