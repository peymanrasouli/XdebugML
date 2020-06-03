import numpy as np

def Mutate(x):

    y = x.copy()

    ind = np.random.randint(0,len(x))

    y[ind] = 1 - y[ind]

    return y
