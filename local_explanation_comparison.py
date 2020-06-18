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
        'lr': LogisticRegression,
        'gt': GradientBoostingClassifier,
        'rf': RandomForestClassifier,
        'nn': MLPClassifier,

    }

    K_list = {
        'german': 200,
        'compas': 500,
        'adult': 800
    }

    print('Local explanation experiment is running...')

    for dataset_kw in datsets_list:
        print('dataset=',dataset_kw)
        # Reading a data set
        dataset_name, prepare_dataset_fn = datsets_list[dataset_kw]
        dataset = prepare_dataset_fn(dataset_name, path_data)
        # Splitting the data set to train, test, and explain set
        X, y = dataset['X'], dataset['y']

        if dataset_kw == 'adult':
            indices = np.random.choice(range(len(X)),size=10000,replace=False)
            X = X[indices]
            y = y[indices]

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
            exists = os.path.isfile(path_exp + 'local_explanation_comparison_results_%s_%s_%s.csv' %
                                      (dataset_kw, blackbox_name,'K_'+str(K_list[dataset_kw])))
            if exists:
                os.remove(path_exp + 'local_explanation_comparison_results_%s_%s_%s.csv' %
                                      (dataset_kw, blackbox_name,'K_'+str(K_list[dataset_kw])))
            experiment_results = open(path_exp + 'local_explanation_comparison_results_%s_%s_%s.csv' %
                                      (dataset_kw, blackbox_name,'K_'+str(K_list[dataset_kw])), 'a')


            results = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
                      ('similar_ground_truth_ga', 'similar_ground_truth_rnd', 'similar_bb_prediction_ga',
                       'similar_bb_prediction_rnd', 'jaccard_feature_names_ga', 'jaccard_feature_names_rnd',
                       'similar_feature_values_ga', 'similar_feature_values_rnd', 'deviation_n_features_ga',
                       'deviation_n_features_rnd')
            experiment_results.write(results)


            results = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % ('=average(A4:A1000)', '=average(B4:B1000)',
                                                           '=average(C4:C1000)', '=average(D4:D1000)',
                                                           '=average(E4:E1000)', '=average(F4:F1000)',
                                                           '=average(G4:G1000)', '=average(H4:H1000)',
                                                           '=average(I4:I1000)', '=average(J4:J1000)')
            experiment_results.write(results)

            results = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % ('=stdev(A4:A1000)', '=stdev(B4:B1000)',
                                                           '=stdev(C4:C1000)', '=stdev(D4:D1000)',
                                                           '=stdev(E4:E1000)', '=stdev(F4:F1000)',
                                                           '=stdev(G4:G1000)', '=stdev(H4:H1000)',
                                                           '=stdev(I4:I1000)', '=stdev(J4:J1000)')
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
            N = 50
            indices = np.random.choice(range(len(X_anomaly)), size=N, replace=False)

            # Main Loop
            B = 10
            for i,index in zip(range(len(indices)),indices):
                print('Anomaly instance=',i)
                jaccard_feature_names_ga = list()
                similar_feature_values_ga = list()
                deviation_n_features_ga = list()
                similar_ground_truth_ga = list()
                similar_bb_prediction_ga = list()

                jaccard_feature_names_rnd = list()
                similar_feature_values_rnd = list()
                deviation_n_features_rnd = list()
                similar_ground_truth_rnd = list()
                similar_bb_prediction_rnd = list()

                instance2explain = X_anomaly[index]
                prediction_x, bias_x, contribution_x = ti.predict(surrogate, instance2explain.reshape(1, -1))
                _, nbrs_cKNN = cKNN.kneighbors(contribution_x[:, :, np.argmax(prediction_x)].reshape(1, -1))
                nbrs_cKNN = nbrs_cKNN[0]

                # Picking representative samples using GA
                contributions_x = contributions_[nbrs_cKNN]
                rp_ind_ga = RepresentativePick(B, contributions_x, nbrs_cKNN)
                rp_set_ga = X_train[rp_ind_ga]

                # Picking representative samples randomly
                rp_ind_rnd = np.random.choice(range(len(nbrs_cKNN)), size=B, replace=False)
                rp_ind_rnd = nbrs_cKNN[rp_ind_rnd]
                rp_set_rnd = X_train[rp_ind_rnd]

                # Explaining representative set by GA using EXPLAN
                tau = 250
                N_samples = 500
                feature_names_ga = list()
                feature_values_ga = list()
                n_features_ga = list()
                for b in range(B):
                    exp_rp, info_rp = explan.Explainer(rp_set_ga[b],
                                                       blackbox,
                                                       dataset,
                                                       N_samples=N_samples,
                                                       tau=tau)

                    rule_EXPLAN = exp_rp[1]
                    feature_names_ga.append(list(rule_EXPLAN.keys()))
                    feature_values_ga.append(rule_EXPLAN)
                    n_features_ga.append(len(list(rule_EXPLAN.keys())))
                    similar_ground_truth_ga.append(int(y_train[anomaly_indices[index]] == y_train[rp_ind_ga[b]]))
                    similar_bb_prediction_ga.append(int(pred_train[anomaly_indices[index]] == pred_train[rp_ind_ga[b]]))

                # Explaining representative set by Random using EXPLAN
                tau = 250
                N_samples = 500
                feature_names_rnd = list()
                feature_values_rnd = list()
                n_features_rnd = list()
                for b in range(B):
                    exp_rp, info_rp = explan.Explainer(rp_set_rnd[b],
                                                       blackbox,
                                                       dataset,
                                                       N_samples=N_samples,
                                                       tau=tau)

                    rule_EXPLAN = exp_rp[1]
                    feature_names_rnd.append(list(rule_EXPLAN.keys()))
                    feature_values_rnd.append(rule_EXPLAN)
                    n_features_rnd.append(len(list(rule_EXPLAN.keys())))
                    similar_ground_truth_rnd.append(int(y_train[anomaly_indices[index]] == y_train[rp_ind_rnd[b]]))
                    similar_bb_prediction_rnd.append(int(pred_train[anomaly_indices[index]] == pred_train[rp_ind_rnd[b]]))

                # Calculating explanation comparison metrics
                for i in range(0, B):
                    for ii in range(i, B):

                        if len(feature_names_ga) > ii:
                            # Calculating Jaccard similarity between feature names of the predicates of the rules
                            jaccard = len(set(feature_names_ga[i]) & set(feature_names_ga[ii])) / \
                                      len(set(feature_names_ga[i]) | set(feature_names_ga[ii]))
                            jaccard_feature_names_ga.append(jaccard)

                            # Calculating the similarity between feature values of predicates of the rules
                            similarity = [1 if feature_values_ga[i][f] == feature_values_ga[ii][f] else 0
                                          for f in set(feature_names_ga[i]) & set(feature_names_ga[ii])]
                            [similar_feature_values_ga.append(s) for s in similarity]

                            # Calculating the deviation from the number of predicates in the collected rules
                            deviation = np.abs(n_features_ga[i] - n_features_ga[ii])
                            deviation_n_features_ga.append(deviation)

                        if len(feature_names_rnd) > ii:
                            # Calculating Jaccard similarity between feature names of the predicates of the rules
                            jaccard = len(set(feature_names_rnd[i]) & set(feature_names_rnd[ii])) / \
                                      len(set(feature_names_rnd[i]) | set(feature_names_rnd[ii]))
                            jaccard_feature_names_rnd.append(jaccard)

                            # Calculating the similarity between feature values of predicates of the rules
                            similarity = [1 if feature_values_rnd[i][f] == feature_values_rnd[ii][f] else 0
                                          for f in set(feature_names_rnd[i]) & set(feature_names_rnd[ii])]
                            [similar_feature_values_rnd.append(s) for s in similarity]

                            # Calculating the deviation from the number of predicates in the collected rules
                            deviation = np.abs(n_features_rnd[i] - n_features_rnd[ii])
                            deviation_n_features_rnd.append(deviation)

                print('similar_ground_truth_ga  =', np.mean(similar_ground_truth_ga))
                print('similar_ground_truth_rnd =', np.mean(similar_ground_truth_rnd))
                print('similar_bb_prediction_ga  =', np.mean(similar_bb_prediction_ga))
                print('similar_bb_prediction_rnd =', np.mean(similar_bb_prediction_rnd))
                print('jaccard_feature_names_ga  =', np.mean(jaccard_feature_names_ga))
                print('jaccard_feature_names_rnd =', np.mean(jaccard_feature_names_rnd))
                print('similar_feature_values_ga  =', np.mean(similar_feature_values_ga))
                print('similar_feature_values_rnd =', np.mean(similar_feature_values_rnd))
                print('deviation_n_features_ga  =', np.mean(deviation_n_features_ga))
                print('deviation_n_features_rnd =', np.mean(deviation_n_features_rnd))
                print('-------------------------------------------------------------')

                results = '%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f' % (
                            np.mean(similar_ground_truth_ga),
                            np.mean(similar_ground_truth_rnd),
                            np.mean(similar_bb_prediction_ga),
                            np.mean(similar_bb_prediction_rnd),
                            np.mean(jaccard_feature_names_ga),
                            np.mean(jaccard_feature_names_rnd),
                            np.mean(similar_feature_values_ga),
                            np.mean(similar_feature_values_rnd),
                            np.mean(deviation_n_features_ga),
                            np.mean(deviation_n_features_rnd),
                )

                # Writing the information to csv file
                results = '%s\n' % (results)
                experiment_results.write(results)
                experiment_results.flush()

            experiment_results.close()


if __name__ == "__main__":
    main()
