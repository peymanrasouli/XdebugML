import random
def WeightedRandomChoice(pop):
    max = sum(p['fitness'] for p in pop)
    pick = random.uniform(0, max)
    current = 0
    for p in pop:
        current += p['fitness']
        if current > pick:
            return p