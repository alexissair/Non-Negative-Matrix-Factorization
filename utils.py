import numpy as np
from scipy.stats import entropy

def kl(a, b) :
    assert a.shape == b.shape
    aprime = a.reshape(-1, 1)
    bprime = b.reshape(-1, 1)
    entr = entropy(aprime, bprime) - np.sum(a) + np.sum(b)
    return entr
