"""
计算聚类得分
TODO: 增加其他的社团检测算法:比如谱聚类，层次聚类，leiden
[1]From Louvain to Leiden: guaranteeing well-connected communities
[2] Pons, P. & Latapy, M. Computing communities in large networks using random walks. Computer
and Information Sciences - ISCIS 284 (2005).
[3]S. Fortunato. Community detection in graphs. Physics Reports, 486(3-5):75{174,
2010.
"""

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score

def compute_score(predict_labels, labels):
    """计算聚类得分"""
    ari = adjusted_rand_score(predict_labels, labels)
    ami = adjusted_mutual_info_score(predict_labels, labels)
    nmi = normalized_mutual_info_score(predict_labels, labels)
    print('-'*30)
    print('ari\tami\tnmi')
    print('{:.4f}  {:.4f}  {:.4f}'.format(ari, ami, nmi))
    return [ari, ami, nmi]