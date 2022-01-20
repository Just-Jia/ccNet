import numpy as np
import matplotlib.pyplot as plt

def statistics_old(adm):
    """
    计算网络的各项统计量
    """
    print(' ===== statistics : ========== ')
    print(' - nodes number   :', num_nodes(adm))
    print(' - edges number   :', num_edges(adm))
#     print(' - degree seq(0-9):', degree_sequence(adm)[0:10] )
    print(' - minimum degree :', min_degree(adm))
    print(' - maximum degree :', max_degree(adm))
    print(' - mean degree    :', mean_degree(adm))
#     K, PK = degree_distribution(adm)
    
def num_nodes(adm):
    """
    计算网络的节点数
    """
    return adm.shape[0]

def num_edges(adm):
    """
    计算网络的总边数
    """
    return int(np.sum(adm)/2)

def degree_sequence(adm):
    """
    计算节点的度序列，返回一个列表，每一项为对应节点的度
    """
    return list( np.sum(adm, axis=0, dtype=int) )

def degree_distribution(adm):
    """
    计算度，及其对应的度分布
    """
    
    print( ' - calculating and ploting degree distribution ... ', end='')
    degree_seq = degree_sequence(adm)
    n = len(degree_seq)
    min_d = min(degree_seq)
    max_d = max(degree_seq)

    K = [i for i in range(min_degree(adm), max_degree(adm)+1)]
    Pk = np.zeros(( max_d), dtype=int)
    for i in range(n):
        ind = degree_seq[i]-1
        Pk[ind] += 1
    Pk = list( Pk[ min_d-1:max_d] / n)
    
    # 绘制直方图
    plt.hist(degree_sequence(adm), bins=100)
    plt.title('Degree Sequence', fontsize=18)
    plt.xlabel('Degree', fontsize=14)
    plt.ylabel('Nodes Number', fontsize=14)
    
    # 绘制度分布图
    fig, ax = plt.subplots()
    ax.scatter(K,Pk, c='red', s=10)
    ax.set_title('Degree Distribution P(k)', fontsize=18)
    ax.set_xlabel('Degree k', fontsize=14)
    ax.set_ylabel('Degree Distribution P(k)', fontsize=14)

    plt.show()
    print('done')
    return K, Pk
    

def min_degree(adm):
    """
    计算节点的最小度
    """
    degree_seq = degree_sequence(adm)
    return min(degree_seq)

def max_degree(adm):
    """
    计算节点的最大度
    """
    degree_seq = degree_sequence(adm)
    return max(degree_seq)

def mean_degree(adm):
    """
    计算节点的平均度
    """
    return (2 * num_edges(adm))/num_nodes(adm)

def statistics(adm, ax=None, c='C0', s=10):
    """
    计算网络的各项统计量
    """

    if not ax is None:
        k, nk = number_k(adm)
        for i in range(len(k)):
            ax.plot([k[i], k[i]],[0, nk[i]], lw=1, c='C1', zorder=1)
        ax.scatter(k, nk, s=s, c=c, zorder=2)
        ax.set_xlabel('Degree $k$')
        ax.set_ylabel("Node number $N_k$ (or P(k)*N)" )
    results = {}
    results['n_nodes'] = num_nodes(adm)
    results['n_edges'] = num_edges(adm)
    results['min_deg'] = min_degree(adm)
    results['max_deg'] = max_degree(adm)
    results['mean_deg'] = mean_degree(adm)

    return results

def number_k(adm):
    """
    计算度，及其对应的度分布
    """
    
    # print( ' - calculating and ploting degree distribution ... ', end='')
    degree_seq = degree_sequence(adm)
    n = len(degree_seq)
    min_d = min(degree_seq)
    max_d = max(degree_seq)

    k = [i for i in range(min_degree(adm), max_degree(adm)+1)]
    nk = np.zeros(( max_d+1), dtype=int)
    for i in range(n):
        ind = degree_seq[i]
        nk[ind] += 1
    nk = list( nk[ min_d:max_d+1] )
    
    return k, nk