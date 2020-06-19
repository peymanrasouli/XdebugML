import os
import numpy as np
from EXPLAN import explan
from EXPLAN.utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from treeinterpreter import treeinterpreter as ti
from representative_pick import RepresentativePick
import warnings
warnings.filterwarnings("ignore")

def main():

    # Defining path of data sets and experiment results
    path = './'
    path_data = path + 'EXPLAN/datasets/'
    path_exp = path + 'experiments/'

    # Defining the list of data sets
    datsets_list = {
        'german': ('german_credit.csv', prepare_german_dataset),
        # 'compas': ('compas-scores-two-years.csv', prepare_compass_dataset),
        # 'adult': ('adult.csv', prepare_adult_dataset)
    }

    # Defining the list of black-boxes
    blackbox_list = {
        'lr': LogisticRegression,
        # 'gt': GradientBoostingClassifier,
        # 'rf': RandomForestClassifier,
        # 'nn': MLPClassifier,

    }

    K_list = {
        'german': 200,
        'compas': 500,
        'adult': 2000
    }

    print('Local explanation experiment is running...')

    for dataset_kw in datsets_list:
        print('dataset=',dataset_kw)
        # Reading a data set
        dataset_name, prepare_dataset_fn = datsets_list[dataset_kw]
        dataset = prepare_dataset_fn(dataset_name, path_data)
        # Splitting the data set to train, test, and explain set
        X, y = dataset['X'], dataset['y']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for blackbox_name in blackbox_list:
            print('blackbox=',blackbox_name)
            # Creating and training black-box
            BlackBoxConstructor = blackbox_list[blackbox_name]
            blackbox = BlackBoxConstructor()
            blackbox.fit(X_train, y_train)
            pred_test = blackbox.predict(X_test)
            bb_accuracy = accuracy_score(y_test, pred_test)
            print('blackbox accuracy=', bb_accuracy)

            dfX2E = build_df2explain(blackbox, X_train, dataset).to_dict('records')

            # Creating/opening a csv file for storing results
            experiment_results = open(path_exp + 'local_explanation_results_%s_%s_%s.csv' %
                                      (dataset_kw, blackbox_name,'K_'+str(K_list[dataset_kw])), 'a')

            # Random Forest surrogate model construction
            pred_train = blackbox.predict(X_train)
            surrogate = RandomForestClassifier(n_estimators=200)
            surrogate.fit(X_train, pred_train)
            prediction, bias, contributions = ti.predict(surrogate, X_train)
            contributions_ = np.zeros(np.shape(X_train))
            for i in range(len(contributions_)):
                contributions_[i,:] = contributions[i,:,np.argmax(prediction[i])]

            # Find anomaly instances in test set
            anomaly_indices = np.where(pred_train != y_train)[0]
            X_anomaly = X_train[anomaly_indices]

            # Creating KNN models for feature values and contribution values
            K = K_list[dataset_kw]
            cKNN = NearestNeighbors(n_neighbors=K).fit(contributions_)

            # Selecting instances to explain
            index = 0
            instance2explain = X_anomaly[index]
            prediction_x, bias_x, contribution_x = ti.predict(surrogate, instance2explain.reshape(1, -1))
            _, nbrs_cKNN = cKNN.kneighbors(contribution_x[:, :, np.argmax(prediction_x)].reshape(1, -1))
            nbrs_cKNN = nbrs_cKNN[0]

            # Picking representative samples
            B = 10
            contributions_x = contributions_[nbrs_cKNN]
            rp_ind= RepresentativePick(B, contributions_x, nbrs_cKNN)
            rp_set = X_train[rp_ind]

            # Explaining isntance2explain using EXPLAN
            tau = 500
            N_samples = 5000
            exp_EXPLAN, info_EXPLAN = explan.Explainer(instance2explain,
                                                       blackbox,
                                                       dataset,
                                                       N_samples=N_samples,
                                                       tau=tau)

            # Reporting the results
            print('\n')
            print('instance2explain =', str(dfX2E[index]))
            print('ground-truth =', str(y_train[anomaly_indices[index]]))
            print('blackbox-pred =', str(pred_train[anomaly_indices[index]]))
            print('explanation = %s' % exp_EXPLAN[1])
            print('\n')

            # Writing the information to csv file
            results = '%s,%s\n%s,%s\n%s,%s\n%s,%s\n\n' % ('instance2explain =',str(dfX2E[index]),
                                                          'ground-truth =',str(y_train[anomaly_indices[index]]),
                                                          'blackbox-pred =',str(pred_train[anomaly_indices[index]]),
                                                          'explanation =',str(exp_EXPLAN[1]))
            experiment_results.write(results)


            # Explaining representative set using EXPLAN
            tau = 500
            N_samples = 5000
            for b in range(B):
                exp_EXPLAN, info_EXPLAN = explan.Explainer(rp_set[b],
                                                           blackbox,
                                                           dataset,
                                                           N_samples=N_samples,
                                                           tau=tau)

                # Reporting the results
                dfx = dfX2E[rp_ind[b]]
                print('representative %s = %s' % (b,dfx))
                print('ground-truth  = %s' % y_train[rp_ind[b]])
                print('blackbox-pred = %s' % pred_train[rp_ind[b]])
                print('explanation = %s' % exp_EXPLAN[1])
                print('\n')

                # Writing the information to csv file
                results = '%s,%s\n%s,%s\n%s,%s\n%s,%s\n\n' % ('representaive '+ str(b) + ' =', str(dfx),
                                                              'ground-truth =', str(y_train[rp_ind[b]]),
                                                              'blackbox-pred =', str(pred_train[rp_ind[b]]),
                                                              'explanation =',str(exp_EXPLAN[1]))
                experiment_results.write(results)

            results = '\n'
            experiment_results.write(results)
            experiment_results.flush()
            experiment_results.close()


if __name__ == "__main__":
    main()
