import numpy as np

def kl(a, b) :
    assert a.shape == b.shape
    s = 0
    eps = 1e-15
    a += eps
    b += eps
    for i in range(a.shape[0]):
        for j in range(a.shape[1]) :
            s += a[i, j] * np.log(a[i, j]/b[i,j]) - a[i,j] + b[i,j]
    return s
