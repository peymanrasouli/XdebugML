import numpy as np
from GA.fitness_function import FitnessFunction
from GA.weighted_random_choice import WeightedRandomChoice
from GA.cross_over import Crossover
from GA.mutate import Mutate
from matplotlib import pyplot as plt

def ga(W, Wb, B, nPop=100, MaxIt=100):

    ## Problem definition
    VarSize = len(W)   # Decision Variables Matrix Size

    ## GA parameters
    pc = 0.8    # Crossover Percentage
    pm = 0.3    # Mutation Percentage
    nc = 2 * round(pc * nPop / 2) # Number of Offsprings
    nm = round(pm * nPop) # Number of Mutants

    ## Initialization
    empty_individual = {
                        'position':[],
                        'fitness':[],
                        'out':[]
                        }

    pop = [empty_individual.copy() for _ in range(nPop)]

    for i in range(nPop):
        # Initialize positions
        pop[i]['position'] = np.zeros(VarSize).astype(int)
        pop[i]['position'][np.random.randint(VarSize, size=B)] = 1
        # Evaluation
        pop[i]['fitness'], pop[i]['out'] = FitnessFunction(pop[i]['position'], W, Wb, B)

    # Sort population
    Fitness = [pop[i]['fitness'] for i in range(len(pop))]
    order = np.argsort(Fitness)[::-1]
    pop = [pop[i] for i in order]

    # Store best solution
    bestSol = pop[0]

    # Array to Hold Best Fitness Values
    bestFitness = np.zeros(MaxIt)

    # Store worst fitness
    worstFitness = pop[-1]['fitness']

    ## Main loop
    for it in range(MaxIt):

        # Crossover
        popc_1 =  [empty_individual.copy() for _ in range(int(nc/2))]
        popc_2 =  [empty_individual.copy() for _ in range(int(nc/2))]

        for k in range(int(nc/2)):

            p1 = WeightedRandomChoice(pop)
            p2 = WeightedRandomChoice(pop)

            # Apply crossover
            popc_1[k]['position'], popc_2[k]['position'] = Crossover(p1['position'], p2['position'])

            # Evaluate offsprings
            popc_1[k]['fitness'], popc_1[k]['out'] = FitnessFunction(popc_1[k]['position'], W, Wb, B)
            popc_2[k]['fitness'], popc_2[k]['out'] = FitnessFunction(popc_2[k]['position'], W, Wb, B)

        popc = popc_1 + popc_2

        # Mutation
        popm = [empty_individual.copy() for _ in range(nm)]

        for k in range(nm):
            # Select Parents
            i = np.random.randint(0,nPop,1)[0]
            p = pop[i]

            # Apply mutation
            popm[k]['position']= Mutate(p['position'])

            # Evaluate mutant
            popm[k]['fitness'], popm[k]['out'] = FitnessFunction(popm[k]['position'], W, Wb, B)

        # Create merged population
        pop = pop+popc+popm

        # Sort population
        Fitness = [pop[i]['fitness'] for i in range(len(pop))]
        order = np.argsort(Fitness)[::-1]
        pop = [pop[i] for i in order]

        # update worst case
        worstFitness = np.minimum(worstFitness,pop[-1]['fitness'])

        # Truncation
        pop = [pop[i] for i in range(nPop)]

        # Store best solution ever found
        bestSol = pop[0]

        # Store best fitness ever found
        bestFitness[it] = bestSol['fitness']

        # Show iteration information
        # print('Iteration=',it,'--','Best fitness=',bestFitness[it])

    return np.where(bestSol['position']==1)



