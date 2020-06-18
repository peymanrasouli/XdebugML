import numpy as np
from GA import ga

def RepresentativePick(B, contributions_x, nbrs_cKNN):

    # Finding most important features
    contributions_x = np.abs(contributions_x)
    con_sorted = np.argsort(-contributions_x)
    con_sorted = con_sorted[:,:int(contributions_x.shape[1]/2)]
    feature_list = np.unique(con_sorted)

    # Creating weight matrix of contributions
    W = np.zeros([con_sorted.shape[0],max(feature_list)+1])
    for i in range(len(W)):
        W[i,con_sorted[i]] = contributions_x[i,con_sorted[i]]
    Wb = np.zeros(np.shape(W)).astype(int)
    Wb[np.where(W>0)] = 1

    # Representative pick using genetic algorithm
    ga_ind = ga.ga(W, Wb, B, nPop=200, MaxIt=50)
    rp_ind = nbrs_cKNN[ga_ind]

    return rp_ind