import numpy as np
from scipy.stats import variation

def FitnessFunction(R, W, Wb, B):
    W_ = W.copy()
    Wb_ = Wb.copy()

    W_ = W_[R]
    Wb_ = Wb_[R]

    values = np.sum(W_,axis=0)

    weights = variation(Wb_, axis=0)
    weights[np.isnan(weights)] = 0

    fitness = np.dot(values,weights)

    if len(np.unique(R)) != B:
        fitness = 0

    out = {
        'weights': weights,
        'values': values
    }

    return fitness, out

