from math import sqrt
from numpy import argsort

from .pearson import pearson


def spearman(x, y):
    """
    Compute the Spearman rank correlation coefficient of two lists with the same lenght.
    If none list have tied value, use formula (1)
        r = 1 - 6 * sum (U-V)^2 / n*(n^2 - 1)
    Otherwise, use formula (2)
        r = S_uv / sqrt( S_uu * S_vv )

    intput:
        x - a list
        y - a list, have the same lenght with x
    output:
        result - a number, Pearson correlation coefficient
    """
    
    if type(x) != list or type(y) != list:
        raise Exception('Two inputs must be two lists!')
    elif len(x) != len(y):
        raise Exception('Two lists must have the same length!')

    # check if have tied value
    # if both do not have tied value, apply formula (1), otherwise formula (2)
    if len(set(x)) == len(x) and len(set(y)) == len(y):
        q = lambda n: map(lambda val: sorted(n).index(val) + 1, n)
        d = sum(map(lambda x, y: (x - y) ** 2, q(x), q(y)))
        return 1.0 - 6.0 * d / float(len(x) * (len(y) ** 2 - 1.0))
    else:
        return pearson(rank(x), rank(y))
        
        
        
def rank(x):
    """
    Compute the ranks corresponding to each item in x.
    """
    l = len(x)
    ind = argsort(x)
    sortedx = [x[ind[i]] for i in range(l)]
    rankx = [i+1 for i in range(l)]
    
    # deal with the rank of tied value
    i = 0
    while i < l - 1:
        see = 1
        while sortedx[i] == sortedx[i+see]:
            see = see + 1
            if i + see >= l:
                break
        if see > 1:
            mean = float( sum(rankx[i:i+see]) / see )
            for j in range(see):
                rankx[i+j] = mean
        i = i + see

    r = [0 for i in range(l)]
    for i in range(l): r[ind[i]] = rankx[i]
    return r