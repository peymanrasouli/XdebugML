import numpy as np

def Mutate(x, VarMin, VarMax):

    y = x.copy()

    ind = np.random.randint(0,len(x))

    mu_value = np.random.randint(VarMin, VarMax, 1)

    y[ind] = mu_value

    return y
