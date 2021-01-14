"""
利用第三方包ripser来计算持续同调
"""
import numpy as np
from ripser import ripser
from scipy.spatial.distance import pdist, squareform

import persim



def homology(data, knn=3, verbose=0):
    """
    Compute Non-uniform Persistent Homology.
    Input:
        data -   n*m np.array, the data we want to compute
        knn -    integer, the refer neighbors number
        verbose - verbosity, 0 or 1
    Output:
        intervals - list of list
    """
    
    n = data.shape[0]
    
    # distance matrix
    matrix_dist = squareform(pdist(data, metric='euclidean'))

    # calculate growth rates matrix
    matrix_nn = np.sort(matrix_dist)    # nearest neighbors matrix
    if knn==0:
        growth_rates = np.ones((1, n))
    else:
        growth_rates = np.mean(matrix_nn[:, range(1, knn+1)], axis=-1)
        growth_rates = np.reshape(growth_rates, (1,n))

    # calculate the time at which point pairs connect
    matrix_grow = growth_rates + np.transpose(growth_rates)
    times = matrix_dist / matrix_grow
    
    intervals = ripser(times, distance_matrix=True)['dgms']
    
    return intervals

def plot(ax, intervals, dim=0, gtype='barcode', verbose=0):
    """
    Plot the Persistent Homology.
    Input:
        gtype -   string. 'barcode', plot barcode graph.
                    'diagram', plot persistence diagram.
    """
    
    if gtype == 'barcode':
        y = [i+1 for i in range(len(intervals[dim]))]
        x = np.array(intervals[dim])

        # if dim equal 0, modified the inf
        if dim==0:
            x[x.shape[0]-1, x.shape[1]-1] = x[x.shape[0]-2, x.shape[1]-1] + 0.1

        out = ax.hlines(y, x[:,0], x[:,1], color='b', lw=1)
        ax.set_xlabel('persistent time')
        ax.set_ylabel('dim-'+str(dim))
    
        return out
    
    elif gtype == 'diagram':
        persim.plot_diagrams(ax=ax, diagrams=intervals)
        
    
    
    
    