import numpy as np
from scipy.stats import norm
from warnings import warn

def wilcoxon(lista, listb, H1='b>a', verbose=1):
    """
    Uing wilcoxon test to test two samples whether have the same distribution.
    Input:
        lista -  a list, containing sample A
        listb -  a list, containing sample B
        H1 -     a string, describing the alternative assumption, it can be
                'a>b', 'a<b', 'b>a', 'b<a'
        verbose - a number, verbosity, 0 or 1
    output:
        u -     a number, test characteristic
        p -     a number, p-value
    """
#     H1 = 'a<b'

#     # sample A and B
#     setA = [104, 110, 106, 113, 115, 111, 102, 128, 110, 117]
#     setB = [94, 103, 114, 126, 95, 102, 100, 98, 103, 116, 105, 107]
    setA = lista
    setB = listb

    # sample size of A and B
    na = len(setA)
    nb = len(setB)

    # combine sample labels and values
    # 0 - represent sample A, 1 - sample B
    A = [[0., setA[i]] for i in range(na)]
    B = [[1., setB[i]] for i in range(nb)]

    table = np.vstack((np.array(A), np.array(B)))

    # sort table based the sample values, get a new talble
    ind = np.lexsort(table.T)
    sorted_table = np.array([ table[ind[i],:] for i in range(len(ind)) ])

    # add rank for each sample value
    rank = np.array([[i+1] for i in range(na+nb)])
    M = np.hstack((sorted_table, rank))

    # here M is a 2-d array of the form, 
    # [sample labels, sample value, rank]

    # deal with the tied values
    i = 0
    while i < na+nb-1:
        see = 1
        while M[i, 1]==M[i+see, 1]:
            see = see + 1
            if  i+see >= na+nb:
                break
        if see > 1:
            s = 0
            for j in range(see):
                s = s + M[i+j, 2]
            mean = float(s / see)
            for j in range(see):
                M[i+j, 2] = mean
        i = i + see

#     print(M)

    # compute rank sum of sample A and B
    Ta = 0.0
    Tb = 0.0
    for i in range(na+nb):
        if M[i, 0] == 0:
            Ta = Ta + M[i, 2]
        else:
            Tb = Tb + M[i, 2]

    # compute mean value (mu), and standard variance (sigma)
    mu = na * (na + nb + 1) / 2
    sigma = (mu * nb / 6) ** 0.5
    
    if verbose == 1:
        print(' - mu = ', mu)
        print(' - sigma = ', sigma)

    # H0: sample B and A share the same distribution
    # H1: values in sample B is large than values in A

    # This is a one-side test, so u should minus 0.5

    # compute the test characteristic according to the alternatve assumption 
    if H1 == 'a>b':
        u  = (Ta - mu - 0.5) / sigma
    elif H1 == 'a<b':
        u  = (Ta - mu + 0.5) / sigma
    elif H1 == 'b>a':
        u  = (Tb - mu - 0.5) / sigma
    elif H1 == 'b<a':
        u  = (Tb - mu + 0.5) / sigma
    else:
        raise Exception("Wrong parameter: must input alternative assumption!")

    meanA = sum(setA) / na
    meanB = sum(setB) / nb


    if meanA > meanB and H1 in ['a<b', 'b>a']:
        warn('mean A is larger than B, but the H1 is "a < b" or "b > a" ')
    elif H1 in ['a>b', 'b<a']:
        warn('mean A is smaller than B, but the H1 is "a > b" or "b < a" ')

    p = 1 - norm.cdf(u)
    
    if verbose == 1:
        print(' - test char. u = ', u)
        print(' - p-value = ', p)
    return u, p