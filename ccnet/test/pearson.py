from math import sqrt

def pearson(x, y):
    """
    Compute the Pearson correlation coefficient of two lists with the same lenght.
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
    
    q = lambda n: len(n) * sum(map(lambda i: i ** 2, n)) - (sum(n) ** 2)
    return (len(x) * sum(map(lambda a: a[0] * a[1], zip(x, y))) - sum(x) * sum(y)) / sqrt(q(x) * q(y))