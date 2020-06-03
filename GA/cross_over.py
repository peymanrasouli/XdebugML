import numpy as np
def Crossover(x1,x2):

    point = int(len(x1)/2)

    offspring1 = np.zeros(len(x1))
    offspring2 = np.zeros(len(x2))

    offspring1[:point] = x1[:point]
    offspring1[point:] = x2[point:]

    offspring2[:point] = x2[:point]
    offspring2[point:] = x1[point:]

    return offspring1, offspring2