import numpy as np
from scipy.stats import variation

def FitnessFunction(s, W, Wb, B):
    W_ = W.copy()
    Wb_ = Wb.copy()

    W_ = W_[np.where(s == 1)]
    Wb_ = Wb_[np.where(s == 1)]

    values = np.sum(W_,axis=0)

    weights = variation(Wb_, axis=0)
    weights[np.isnan(weights)] = 0

    fitness = np.dot(values,weights)

    if sum(s) != B:
        fitness = 0

    out = {
        'weights': weights,
        'values': values
    }
    return fitness, out


