# 计算差异表达基因
import numpy as np
from scipy.sparse import csr_matrix
from sknetwork.path import shortest_path
from scipy import stats

def tags(wadm, data, ordering, gene_names, source=None, target=None, verbose=0):
    """find trajectory assiciated genes
    Parameters:
        wadm:
            weighted adjacency matrix,
        data:
            expression matrix, cell * gene
        source:
            root cell
        target:
            target cell
    Return:
        sorted dict, key is gene, value is (pearson correlation, p-value)
    """
    
    if source is None or target is None:
        raise Exception("source and target cannot be None")
    
    # path node, corresponding distance 
    path = shortest_path(csr_matrix(wadm), sources=source, targets=target)
    path_dist = [ordering[node] for node in path]
    
    if verbose==1:
        print('path:', path)
        print('dist:', path_dist)
    
    # test the correlation between path distance and expression
    path_data = data[np.ix_(path, )]
    
    pearsons = {}
    for i in range(path_data.shape[1]):
        c, p = stats.pearsonr(path_dist, path_data[:,i])
        pearsons[gene_names[i]] = (c,p)

    # 处理nan的情况，当表达值为常数会出现该情况
    for key, value in pearsons.items():
        if np.isnan(value[0]):
            pearsons[key] = (0,1)
            
    # 第四步，找出相关性最大的基因
    pearsons= sorted(pearsons.items(), key=lambda d:d[1][0], reverse = True)
    
    return path, pearsons