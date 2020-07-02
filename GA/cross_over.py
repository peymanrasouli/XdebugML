import numpy as np

def Crossover(x1,x2):

    point = np.random.randint(0, len(x1))

    offspring1 = np.zeros(len(x1))
    offspring2 = np.zeros(len(x2))

    offspring1[:point] = x1[:point]
    offspring1[point:] = x2[point:]

    offspring2[:point] = x2[:point]
    offspring2[point:] = x1[point:]

    return offspring1.astype(int), offspring2.astype(int)
