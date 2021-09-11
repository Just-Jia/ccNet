import numpy as np
from scipy.spatial.distance import pdist, squareform

def sigmoid(adm, data, k=4, metric='euclidean', c=0.5, h=1.5, s=30):
    """
    Weighting the edges by Sigmoid function.
    
    Parameters
    ----------
    
    adm: 2d np.array
        adjacency matrix of network
    data:
        data 
    k: int
        number of neighbors for reference
    c: float
        center of sigmoid function
    h: float
        height of sigmoid function
    s: folat
        slope of sigmoid function
        
    Returns
    -------
    
    wadm: 2d np.array
        weighted adjacency matrix
    """

    n = data.shape[0]

    # distance matrix
    matrix_dist = squareform(pdist(data, metric=metric))

    # calculate growth rates matrix
    matrix_nn = np.sort(matrix_dist)    # nearest neighbors matrix
    if k==0:
        growth_rates = np.ones((1, n))
    else:
        growth_rates = np.mean(matrix_nn[:, range(1, k+1)], axis=-1)
        growth_rates = np.reshape(growth_rates, (1,n))

    # calculate the time at which point pairs connect
    matrix_grow = growth_rates + np.transpose(growth_rates)
    times = matrix_dist / matrix_grow

    wadm = h / (1 + np.exp(s*(times - c)))
    wadm = wadm * adm
    
    return wadm