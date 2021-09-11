"""
TODO: 增加其他的社团检测算法:比如谱聚类，层次聚类，leiden
[1]From Louvain to Leiden: guaranteeing well-connected communities
[2] Pons, P. & Latapy, M. Computing communities in large networks using random walks. Computer
and Information Sciences - ISCIS 284 (2005).
[3]S. Fortunato. Community detection in graphs. Physics Reports, 486(3-5):75{174,
2010.
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
from numpy import linalg as LA

from ..network import *

def sortdata(data):
    """
    将数据进行排队，并返回新的数据
    """
    print(' - sorting data')
    print(data.shape)
    net = network_test(data, knn=3, time=0.6,solid=False, connected=True)
    
    # 计算net的laplacian
    D = np.diag(np.sum(net, axis=-1))
    L = D - net

    # 计算laplacian 最小非零特征值对应的的特征向量，
        
    # 计算特征值w,和特征对应的特征向量v
    w, v = LA.eig(L)
#     print(w.shape, type(w))
    ind = np.argsort(w)
    vector = v[:,ind[1]]
#         print(vector)
        
    # 根据特征向量对节点进行排序
    orders = np.argsort(vector)
#         print(orders)
    new_data = np.array([ data[orders[i],:] for i in range(len(orders)) ])
    return new_data


def compute_distm(data, labels=None):
    
    if labels is None:
        new_data = data.copy()
    else:
        # 得到标签列表
        lab_set = list(set(labels))
        nlab = len(lab_set)
        print('lab_set = ',lab_set)
        n = len(labels)
        m = data.shape[1]

        pos_c = np.zeros((nlab,m))
        for i in range(nlab):
            pos_sum = np.zeros((1,m))
            num = 0
            for j in range(n):
                if labels[j]==lab_set[i]:
                    pos_sum = pos_sum + data[j,:]
                    num = num + 1
            pos_c[i,:] = pos_sum / num
        net = network_test(pos_c, method='nen', knn=3, time=0.6, connected=True)
        # 计算net的laplacian
        D = np.diag(np.sum(net, axis=-1))
        L = D - net
        print(L)
        # 计算laplacian 最小非零特征值对应的的特征向量，
        # 计算特征值w,和特征对应的特征向量v
        w, v = LA.eig(L)
        print(w.shape, type(w))
        ind = np.argsort(w)
        vector = v[:,ind[1]]
        print(vector)
        # 根据特征向量对节点进行排序
        orders = np.argsort(vector)
        print(orders)
        # 标签顺序
        labelorder = [lab_set[orders[i]] for i in range(nlab)]
        # !!!!!!!!!!!!  对已经进行排序的标签进行排序
    #     labelorder = np.sort(labelorder)
        print(labelorder)

        new_data = np.empty((0,m))
        for i in range(nlab):
            temp_data = []
            for j in range(n):
    #             print(i,j)
                if labels[j] == labelorder[i]:
                    temp_data.append(data[j,:])
            temp_data = np.array(temp_data)
            temp_newdata = sortdata(temp_data)
            new_data = np.vstack(( new_data, temp_newdata ))
    #                 print(labels[j], labelorder[i])
#                     new_data.append(data[j,:])
#         new_data1 = np.array(new_data)

    # new_data = np.array([data[j,:] for i in range(nlab) for j in range(n) if labels[j]==labelorder[i]])
    # 计算距离矩阵
#     from scipy.spatial.distance import pdist, squareform
    matrix_dist = squareform(pdist(new_data, metric='euclidean'))
    return matrix_dist

# the old codes
def compute_distm1(data, labels=None):
    """
    根据标签计算距离矩阵
    """
    if labels is None:
        new_data = data
    else:
        # 对数据根据标签进行排序
        order = np.lexsort(np.hstack((data,np.array([labels]).T)).T)
        # 得到排序后的数据
        new_data = np.array([data[order[i],:] for i in range(data.shape[0]) ])
    # 计算距离矩阵
    matrix_dist = squareform(pdist(new_data, metric='euclidean'))
    
    return matrix_dist