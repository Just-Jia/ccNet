"""
TODO: 增加其他的社团检测算法:比如谱聚类，层次聚类，leiden
[1]From Louvain to Leiden: guaranteeing well-connected communities
[2] Pons, P. & Latapy, M. Computing communities in large networks using random walks. Computer
and Information Sciences - ISCIS 284 (2005).
[3]S. Fortunato. Community detection in graphs. Physics Reports, 486(3-5):75{174,
2010.
"""

import matplotlib.pyplot as plt

def plot_distm(ax, distm):
    """
    绘制距离矩阵
    """
    # 画图
    im = ax.imshow(distm)
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set_title('distance matrix')
    return im