from math import sqrt
from itertools import combinations

def kendall(x, y):
    assert len(x) == len(y) > 0
    c = 0 #concordant count
    d = 0 #discordant count
    t = 0 #tied count
    for (i, j) in combinations(range(len(x)), 2):
        s = (x[i] - x[j]) * (y[i] - y[j])
        if s:
            c += 1
            d += 1
            if s > 0:
                t += 1
            elif s < 0:
                t -= 1
        else:
            if x[i] - x[j]:
                c += 1
            elif y[i] - y[j]:
                d += 1
    return t / sqrt(c * d)