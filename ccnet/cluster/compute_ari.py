"""
TODO: 增加其他的社团检测算法:比如谱聚类，层次聚类，leiden
[1]From Louvain to Leiden: guaranteeing well-connected communities
[2] Pons, P. & Latapy, M. Computing communities in large networks using random walks. Computer
and Information Sciences - ISCIS 284 (2005).
[3]S. Fortunato. Community detection in graphs. Physics Reports, 486(3-5):75{174,
2010.
"""

from sklearn.metrics.cluster import adjusted_rand_score

def compute_ari(labels1, labels2):
    """
    Compute the Adjusted Rand Index
    """
    return adjusted_rand_score(labels1, labels2)