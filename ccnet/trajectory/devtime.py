"""
根据计算出的伪时间以及采样时间，来推断发育时间
"""

def median(sample):
    """
    Find the median of sample, which must be a list.
    """
    n = len(sample)
    s = sorted(sample)
    i = int(n/2)
    if n % 2 == 1:
        return s[i]
    else:
        return (s[i-1] + s[i]) * 0.5
    
def devtime(ordering, labels, verbose=0):
    """
    Compute the develop time for each cell.
    注意：可能会有bug，这里的取样时间必须从1开始
    """

#     infertime = [0, 0, 0, 0, 0]
#     sampletime = [0, 1, 2]
#     medians = [0, 0.9, 2]

    n = len(ordering)
    
    infertime = [0 for i in range(n)]
    
    print('Warning: the sampling time must start with 1')
    
    print(' - reshaping the sampling time ... ', end='')
    sampletime=sorted(list(set(labels)))
    sampletime.insert(0, 0)
    print('done')
    
    print(' - new sampling time =', sampletime)
    
    print(' - computing medians ... ', end='')
    medians = []
    medians.append(0)
    for i in range(1, len(sampletime)):
        cells = [ordering[j] for j in range(n) if labels[j]==sampletime[i]]
        medians.append(median(cells))
    print('done')
    print(' - the medians =', medians)
        
    print(' - scaling the time segmentally ... ', end='')
    for i in range(n):
#         print(' - i =', i)
        j = 1
        flag = True
        while flag:
            if ordering[i] <= medians[j]:
                infertime[i] = sampletime[j-1] + (ordering[i] - medians[j-1]) / (medians[j] - medians[j-1])
                flag = False
            elif ordering[i] > medians[-1]:
                infertime[i] = sampletime[-2] + (ordering[i] - medians[-2]) / (medians[-1] - medians[-2])
                flag = False
            else:
                j += 1
#             print(infertime, flush=True)
    print('done')
            
    return infertime