"""
TODO: 增加其他的社团检测算法:比如谱聚类，层次聚类，leiden
[1]From Louvain to Leiden: guaranteeing well-connected communities
[2] Pons, P. & Latapy, M. Computing communities in large networks using random walks. Computer
and Information Sciences - ISCIS 284 (2005).
[3]S. Fortunato. Community detection in graphs. Physics Reports, 486(3-5):75{174,
2010.
"""
from .odv import *
from .louvain import *
from .leiden import *

import numpy as np

from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

from sklearn.metrics.cluster import adjusted_rand_score

import traceback

def clustering(adm, method='odv', partition=None, ifcheck=True, quantity='modularity', resolution=0.05 ,n_iters=1):
    """
    Operate clustering analysis.
    """
    return community_detect(adm, method=method, partition=partition, ifcheck=ifcheck, quantity=quantity, resolution=resolution, n_iters=n_iters)

def community_detect(adm, method='louvain', partition=None, ifcheck=True, quantity='modularity', resolution=0.05, n_iters=1):
    """
    Perform community detection for complex network.
    input:
        adm -       n*n np.array, represent adjacent matrix of network.
        method -    string, the community detection algorithm used here.
                    Default 'louvain'.
    output:
        labels -    list, containing cluster labels coresponding to each point.
                    It's lenght equals the node number, namely the order of 
                    'adm'.
    """
    print(' - detecting communities ... ', end='')

    if method=='odv':
        labels = odv(net=adm, partition=partition, ifcheck=ifcheck, random=True, n_iters=n_iters)
        # odv 算法的结果是一个字典，key is node, value is community
        # labels = [node_labels[node] for node in sorted(node_labels.keys())]
    elif method=='louvain':
        labels = louvain(adm, n_iters=n_iters)
    elif method=='leiden':
        labels = leiden(adm, quantity=quantity, resolution=resolution, n_iters=n_iters)
    else:
        raise Exception("Wrong parameter: method must be one of louvain or odv!")

    print(' done')
    return labels

def compute_ari(labels1, labels2):
    """
    Compute the Adjusted Rand Index
    """
    
    return adjusted_rand_score(labels1, labels2)


def compute_distm(data, labels=None):
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

def plot_distm(ax, distm):
    """
    绘制距离矩阵
    """
    # 画图
    im = ax.imshow(distm)
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set_title('distance matrix')
    return im