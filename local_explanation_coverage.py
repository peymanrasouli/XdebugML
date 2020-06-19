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
        'compas': ('compas-scores-two-years.csv', prepare_compass_dataset),
        'adult': ('adult.csv', prepare_adult_dataset)
    }

    # Defining the list of black-boxes
    blackbox_list = {
        # 'lr': LogisticRegression,
        'gt': GradientBoostingClassifier,
        # 'rf': RandomForestClassifier,
        # 'nn': MLPClassifier,

    }

    K_list = {
        'german': 200,
        'compas': 500,
        'adult': 2000
    }

    # Hit evaluation function
    def hit_outcome(x, y):
        return 1 if x == y else 0

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
            print('\n')

            # Creating/opening a csv file for storing results
            exists = os.path.isfile(path_exp + 'local_explanation_coverage_results_%s_%s_%s.csv' %
                                      (dataset_kw, blackbox_name,'K_'+str(K_list[dataset_kw])))
            if exists:
                os.remove(path_exp + 'local_explanation_coverage_results_%s_%s_%s.csv' %
                                      (dataset_kw, blackbox_name,'K_'+str(K_list[dataset_kw])))
            experiment_results = open(path_exp + 'local_explanation_coverage_results_%s_%s_%s.csv' %
                                      (dataset_kw, blackbox_name,'K_'+str(K_list[dataset_kw])), 'a')

            results = '%s,%s,%s,%s,%s,%s\n' % ('global_coverage_ga', 'global_coverage_rnd',
                                               'local_coverage_ga','local_coverage_rnd',
                                               'precision_ga', 'precision_rnd')
            experiment_results.write(results)

            results = '%s,%s,%s,%s,%s,%s\n' % ('=average(A4:A1000)', '=average(B4:B1000)',
                                               '=average(C4:C1000)', '=average(D4:D1000)',
                                               '=average(E4:E1000)', '=average(F4:F1000)')
            experiment_results.write(results)

            results = '%s,%s,%s,%s,%s,%s\n' % ('=stdev(A4:A1000)', '=stdev(B4:B1000)',
                                               '=stdev(C4:C1000)', '=stdev(D4:D1000)',
                                               '=stdev(E4:E1000)', '=stdev(F4:F1000)')
            experiment_results.write(results)
            experiment_results.flush()

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
            N = 100
            indices = np.random.choice(range(len(X_anomaly)), size=np.min([len(X_anomaly),N]), replace=False)

            # Main Loop
            B = 10
            for i,index in zip(range(len(indices)),indices):
                print('Anomaly instance=',i)
                local_coverage_ga = list()
                global_coverage_ga = list()
                precision_ga = list()

                local_coverage_rnd = list()
                global_coverage_rnd = list()
                precision_rnd = list()

                instance2explain = X_anomaly[index]
                prediction_x, bias_x, contribution_x = ti.predict(surrogate, instance2explain.reshape(1, -1))
                _, nbrs_cKNN = cKNN.kneighbors(contribution_x[:, :, np.argmax(prediction_x)].reshape(1, -1))
                nbrs_cKNN = nbrs_cKNN[0]

                # Creating data frame of the neighborhood
                dfX = build_df2explain(blackbox, X_train[nbrs_cKNN], dataset)

                # Picking representative samples using GA
                contributions_x = contributions_[nbrs_cKNN]
                rp_ind_ga = RepresentativePick(B, contributions_x, nbrs_cKNN)
                rp_set_ga = X_train[rp_ind_ga]

                # Picking representative samples randomly
                rp_ind_rnd = np.random.choice(range(len(nbrs_cKNN)), size=B, replace=False)
                rp_ind_rnd = nbrs_cKNN[rp_ind_rnd]
                rp_set_rnd = X_train[rp_ind_rnd]

                # Explaining representative set by GA using EXPLAN
                tau = 500
                N_samples = 5000
                for b in range(B):
                    exp_rp, info_rp = explan.Explainer(rp_set_ga[b],
                                                       blackbox,
                                                       dataset,
                                                       N_samples=N_samples,
                                                       tau=tau)

                    rule = exp_rp[1]
                    covered_X = get_covered(rule, dfX.to_dict('records'), dataset)
                    y_x = blackbox.predict(rp_set_ga[b].reshape(1,-1))
                    y_X = blackbox.predict(X_train[nbrs_cKNN])
                    precision_X = [hit_outcome(y, y_x) for y in y_X[covered_X]]
                    precision_X = 0 if precision_X == [] else np.mean(precision_X)

                    global_coverage_ga.append(covered_X)
                    local_coverage_ga.append(len(covered_X)/K)
                    precision_ga.append(precision_X)


                # Explaining representative set by Random using EXPLAN
                tau = 500
                N_samples = 5000
                for b in range(B):
                    exp_rp, info_rp = explan.Explainer(rp_set_rnd[b],
                                                       blackbox,
                                                       dataset,
                                                       N_samples=N_samples,
                                                       tau=tau)

                    rule = exp_rp[1]
                    covered_X = get_covered(rule, dfX.to_dict('records'), dataset)
                    y_x = blackbox.predict(rp_set_rnd[b].reshape(1, -1))
                    y_X = blackbox.predict(X_train[nbrs_cKNN])
                    precision_X = [hit_outcome(y, y_x) for y in y_X[covered_X]]
                    precision_X = 0 if precision_X == [] else np.mean(precision_X)

                    global_coverage_rnd.append(covered_X)
                    local_coverage_rnd.append(len(covered_X) / K)
                    precision_rnd.append(precision_X)

                global_coverage_ga =  len(np.unique(np.concatenate(global_coverage_ga))) / K
                global_coverage_rnd = len(np.unique(np.concatenate(global_coverage_rnd))) / K
                local_coverage_ga =  np.mean(global_coverage_ga)
                local_coverage_rnd = np.mean(global_coverage_rnd)
                precision_ga = np.mean(precision_ga)
                precision_rnd = np.mean(precision_rnd)

                print('global_coverage_ga  =', global_coverage_ga)
                print('global_coverage_rnd =', global_coverage_rnd)
                print('local_coverage_ga   =', local_coverage_ga)
                print('local_coverage_rnd  =', local_coverage_rnd)
                print('precision_ga  =', precision_ga)
                print('precision_rnd =', precision_rnd)
                print('---------------------------------------')

                results = '%.3f,%.3f,%.3f,%.3f,%.3f,%.3f' % (global_coverage_ga, global_coverage_rnd,
                                                             local_coverage_ga, local_coverage_rnd,
                                                             precision_ga, precision_rnd)

                # Writing the information to csv file
                results = '%s\n' % (results)
                experiment_results.write(results)
                experiment_results.flush()

            experiment_results.close()


if __name__ == "__main__":
    main()
