import numpy as np
from GA import ga

def RepresentativePick(B, N_top, contributions_nbrs, nbrs_cKNN):

    # Finding most important features
    contributions_nbrs = np.abs(contributions_nbrs)
    con_sorted = np.argsort(-contributions_nbrs)
    con_sorted = con_sorted[:,:N_top]
    feature_list = np.unique(con_sorted)

    # Creating weight matrix of contributions
    W = np.zeros([con_sorted.shape[0],max(feature_list)+1])
    for i in range(len(W)):
        W[i,con_sorted[i]] = contributions_nbrs[i,con_sorted[i]]
    Wb = np.zeros(np.shape(W)).astype(int)
    Wb[np.where(W>0)] = 1

    # Representative pick using genetic algorithm
    ga_ind = ga.ga(W, Wb, B, nPop=300, MaxIt=100)
    rp_ind = nbrs_cKNN[ga_ind]

    return rp_ind
