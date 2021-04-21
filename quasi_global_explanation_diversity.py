import os
from utils import *
from EXPLAN import explan
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from contribution_extraction import ContributionExtraction
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
        'nn': MLPClassifier,

    }

    K_list = {
        'german': 200,
        'compas': 500,
        'adult': 2000
    }

    print('Quasi-global explanation diversity experiment is running...')

    for dataset_kw in datsets_list:
        print('dataset=',dataset_kw)

        # Reading a data set
        dataset_name, prepare_dataset_fn = datsets_list[dataset_kw]
        dataset = prepare_dataset_fn(dataset_name, path_data)
        
        # Splitting the data set into train and test sets
        X, y = dataset['X'], dataset['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for blackbox_name in blackbox_list:
            print('blackbox=',blackbox_name)

            # Creating and training black-box
            BlackBoxConstructor = blackbox_list[blackbox_name]
            blackbox = BlackBoxConstructor(random_state=42)
            blackbox.fit(X_train, y_train)
            pred_train = blackbox.predict(X_train)
            pred_test = blackbox.predict(X_test)
            bb_accuracy = accuracy_score(y_test, pred_test)
            print('blackbox accuracy=', bb_accuracy)
            print('\n')

            # Creating/opening a csv file for storing results
            exists = os.path.isfile(path_exp + 'quasi_global_explanation_diversity_results_%s_%s_%s.csv' %
                                      (dataset_kw, blackbox_name,'K_'+str(K_list[dataset_kw])))
            if exists:
                os.remove(path_exp + 'quasi_global_explanation_diversity_results_%s_%s_%s.csv' %
                                      (dataset_kw, blackbox_name,'K_'+str(K_list[dataset_kw])))
            experiment_results = open(path_exp + 'quasi_global_explanation_diversity_results_%s_%s_%s.csv' %
                                      (dataset_kw, blackbox_name,'K_'+str(K_list[dataset_kw])), 'a')

            results = '%s,%s,%s,%s,%s,%s,%s,%s\n' % \
                      ('same_pred_anomaly_ga', 'same_pred_anomaly_rnd',
                       'same_pred_ok_ga', 'same_pred_ok_rnd',
                       'jaccard_feature_names_ga', 'jaccard_feature_names_rnd',
                       'deviation_n_features_ga', 'deviation_n_features_rnd')
            experiment_results.write(results)

            results = '%s,%s,%s,%s,%s,%s,%s,%s\n' % \
                      ('=average(A4:A1000)', '=average(B4:B1000)',
                       '=average(C4:C1000)', '=average(D4:D1000)',
                       '=average(E4:E1000)', '=average(F4:F1000)',
                       '=average(G4:G1000)', '=average(H4:H1000)')
            experiment_results.write(results)

            results = '%s,%s,%s,%s,%s,%s,%s,%s\n' % \
                      ('=stdev(A4:A1000)', '=stdev(B4:B1000)',
                       '=stdev(C4:C1000)', '=stdev(D4:D1000)',
                       '=stdev(E4:E1000)', '=stdev(F4:F1000)',
                       '=stdev(G4:G1000)', '=stdev(H4:H1000)')
            experiment_results.write(results)
            experiment_results.flush()

            # Extracting instance-level feature contributions
            explanation_method = 'shapley_sampling_values'  # 'shapley_sampling_values' | 'tree_explainer' | 'tree_interpreter'
            contributions, extractor = ContributionExtraction(blackbox, X_train, method=explanation_method)

            # Finding anomaly instances in the train set
            anomaly_indices = np.where(pred_train != y_train)[0]
            X_anomaly = X_train[anomaly_indices]

            # Creating a KNN model for contribution values
            K = K_list[dataset_kw]
            cKNN = NearestNeighbors(n_neighbors=K).fit(contributions)

            # Selecting instances to explain
            N = 100
            indices = np.random.choice(range(len(X_anomaly)), size=np.min([len(X_anomaly),N]), replace=False)

            # Main Loop
            B = 10
            NF = 5
            for i,index in zip(range(len(indices)),indices):
                print('Anomaly instance=',i)
                jaccard_feature_names_ga = list()
                deviation_n_features_ga = list()
                same_pred_anomaly_ga = list()
                same_pred_ok_ga = list()

                jaccard_feature_names_rnd = list()
                deviation_n_features_rnd = list()
                same_pred_anomaly_rnd = list()
                same_pred_ok_rnd = list()

                instance2explain = X_anomaly[index]
                contribution_x = extractor(instance2explain)
                _, nbrs_cKNN = cKNN.kneighbors(contribution_x.reshape(1, -1))
                nbrs_cKNN = nbrs_cKNN[0]

                # Picking representative samples using GA
                contributions_nbrs = contributions[nbrs_cKNN]
                try:
                    rp_ind_ga = RepresentativePick(B, NF, contributions_nbrs, nbrs_cKNN)
                except Exception:
                    rp_ind_ga = np.random.choice(range(len(nbrs_cKNN)), size=B, replace=False)
                rp_set_ga = X_train[rp_ind_ga]

                # Picking representative samples randomly
                rp_ind_rnd = np.random.choice(range(len(nbrs_cKNN)), size=B, replace=False)
                rp_ind_rnd = nbrs_cKNN[rp_ind_rnd]
                rp_set_rnd = X_train[rp_ind_rnd]

                # Explaining the GA representative set using EXPLAN
                tau = 250
                N_samples = 3000
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
                    sim1 = y_train[anomaly_indices[index]] == y_train[rp_ind_ga[b]]
                    sim2 = y_train[anomaly_indices[index]] != y_train[rp_ind_ga[b]]
                    sim3 = pred_train[anomaly_indices[index]] == pred_train[rp_ind_ga[b]]
                    same_pred_anomaly_ga.append(int(sim1 and sim3))
                    same_pred_ok_ga.append(int(sim2 and sim3))

                # Explaining the Random representative set using EXPLAN
                tau = 250
                N_samples = 3000
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
                    sim1 = y_train[anomaly_indices[index]] == y_train[rp_ind_rnd[b]]
                    sim2 = y_train[anomaly_indices[index]] != y_train[rp_ind_rnd[b]]
                    sim3 = pred_train[anomaly_indices[index]] == pred_train[rp_ind_rnd[b]]
                    same_pred_anomaly_rnd.append(int(sim1 and sim3))
                    same_pred_ok_rnd.append(int(sim2 and sim3))

                # Calculating explanation comparison metrics
                for i in range(0, B):
                    for ii in range(i, B):

                        if len(feature_names_ga) > ii:
                            # Calculating Jaccard similarity between feature names of the predicates of the rules
                            jaccard = len(set(feature_names_ga[i]) & set(feature_names_ga[ii])) / \
                                      len(set(feature_names_ga[i]) | set(feature_names_ga[ii]))
                            jaccard_feature_names_ga.append(jaccard)

                            # Calculating the deviation from the number of predicates in the collected rules
                            deviation = np.abs(n_features_ga[i] - n_features_ga[ii])
                            deviation_n_features_ga.append(deviation)

                        if len(feature_names_rnd) > ii:
                            # Calculating Jaccard similarity between feature names of the predicates of the rules
                            jaccard = len(set(feature_names_rnd[i]) & set(feature_names_rnd[ii])) / \
                                      len(set(feature_names_rnd[i]) | set(feature_names_rnd[ii]))
                            jaccard_feature_names_rnd.append(jaccard)

                            # Calculating the deviation from the number of predicates in the collected rules
                            deviation = np.abs(n_features_rnd[i] - n_features_rnd[ii])
                            deviation_n_features_rnd.append(deviation)

                # Printing the results
                print('same_pred_anomaly_ga  =', np.mean(same_pred_anomaly_ga))
                print('same_pred_anomaly_rnd =', np.mean(same_pred_anomaly_rnd))
                print('same_pred_ok_ga  =', np.mean(same_pred_ok_ga))
                print('same_pred_ok_rnd =', np.mean(same_pred_ok_rnd))
                print('jaccard_feature_names_ga  =', np.mean(jaccard_feature_names_ga))
                print('jaccard_feature_names_rnd =', np.mean(jaccard_feature_names_rnd))
                print('deviation_n_features_ga  =', np.mean(deviation_n_features_ga))
                print('deviation_n_features_rnd =', np.mean(deviation_n_features_rnd))
                print('-------------------------------------------------------------')

                # Writing the results into the csv file
                results = '%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f' % (
                            np.mean(same_pred_anomaly_ga),
                            np.mean(same_pred_anomaly_rnd),
                            np.mean(same_pred_ok_ga),
                            np.mean(same_pred_ok_rnd),
                            np.mean(jaccard_feature_names_ga),
                            np.mean(jaccard_feature_names_rnd),
                            np.mean(deviation_n_features_ga),
                            np.mean(deviation_n_features_rnd)
                )
                results = '%s\n' % (results)
                experiment_results.write(results)
                experiment_results.flush()

            experiment_results.close()

if __name__ == "__main__":
    main()
