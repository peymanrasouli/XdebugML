import numpy as np

def NeighborhoodInfluence(blackbox, extractor, cKNN, fKNN, pKNN, BlackBoxConstructor,
                          X_train, y_train, X_anomaly, n_test=10,perturb_percent=0.75):

    ind = np.random.choice(len(X_anomaly),size=n_test,replace=False)
    X_anomaly = X_anomaly[ind]

    cKNN_influence = list()
    fKNN_influence = list()
    pKNN_influence = list()

    # Main loop
    for i in range(n_test):

        ##########################################################################################
        ########################################## cKNN ##########################################

        ## Achieve neighborhood samples using cKNN method
        contribution_x = extractor(X_anomaly[i])
        _, nbrs_cKNN = cKNN.kneighbors(contribution_x.reshape(1, -1))

        ## Perturb the label of the data inside the neighborhood
        y_train_ = y_train.copy()
        perturbed_idx = np.random.choice(range(len(nbrs_cKNN)),
                                         round(perturb_percent * len(nbrs_cKNN)),
                                         replace=False)
        perturbed_idx = nbrs_cKNN[perturbed_idx].copy()
        y_train_[perturbed_idx] = 1 - y_train_[perturbed_idx]

        # Creating the black box using the perturbed data
        blackbox_ = BlackBoxConstructor(random_state=42)
        blackbox_.fit(X_train, y_train_)

        # Testing influence
        bb_pred = int(blackbox.predict(X_anomaly[i].reshape(1, -1))[0])
        bb_pred_ = int(blackbox_.predict(X_anomaly[i].reshape(1, -1))[0])
        changed = 0 if bb_pred == bb_pred_ else 1
        cKNN_influence.append(changed)

        ##########################################################################################
        ########################################## fKNN ##########################################

        ## Achieve neighborhood samples using fKNN method
        _, nbrs_fKNN = fKNN.kneighbors(X_anomaly[i].reshape(1, -1))
        nbrs_fKNN = nbrs_fKNN[0]

        ## Perturb the label of the data inside the neighborhood
        y_train_ = y_train.copy()
        perturbed_idx = np.random.choice(range(len(nbrs_fKNN)),
                                         round(perturb_percent * len(nbrs_fKNN)),
                                         replace=False)
        perturbed_idx = nbrs_fKNN[perturbed_idx].copy()
        y_train_[perturbed_idx] = 1 - y_train_[perturbed_idx]

        # Creating the black box using the perturbed data
        blackbox_ = BlackBoxConstructor(random_state=42)
        blackbox_.fit(X_train, y_train_)

        # Testing influence
        bb_pred = int(blackbox.predict(X_anomaly[i].reshape(1, -1))[0])
        bb_pred_ = int(blackbox_.predict(X_anomaly[i].reshape(1, -1))[0])
        changed = 0 if bb_pred == bb_pred_ else 1
        fKNN_influence.append(changed)

        ##########################################################################################
        ########################################## pKNN ##########################################

        ## Achieve neighborhood samples using pKNN method
        _, nbrs_pKNN = pKNN.kneighbors(blackbox.predict_proba(X_anomaly[i].reshape(1, -1)))
        nbrs_pKNN = nbrs_pKNN[0]

        ## Perturb the label of the data inside the neighborhood
        y_train_ = y_train.copy()
        perturbed_idx = np.random.choice(range(len(nbrs_pKNN)),
                                         round(perturb_percent * len(nbrs_pKNN)),
                                         replace=False)
        perturbed_idx = nbrs_pKNN[perturbed_idx].copy()
        y_train_[perturbed_idx] = 1 - y_train_[perturbed_idx]

        # Creating the black box using the perturbed data
        blackbox_ = BlackBoxConstructor(random_state=42)
        blackbox_.fit(X_train, y_train_)

        # Testing influence
        bb_pred = int(blackbox.predict(X_anomaly[i].reshape(1, -1))[0])
        bb_pred_ = int(blackbox_.predict(X_anomaly[i].reshape(1, -1))[0])
        changed = 0 if bb_pred == bb_pred_ else 1
        pKNN_influence.append(changed)

    # Return results
    return [np.mean(cKNN_influence),
            np.mean(fKNN_influence),
            np.mean(pKNN_influence)]
