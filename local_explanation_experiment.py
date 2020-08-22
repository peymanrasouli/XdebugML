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
        # 'german': ('german_credit.csv', prepare_german_dataset),
        # 'compas': ('compas-scores-two-years.csv', prepare_compass_dataset),
        'adult': ('adult.csv', prepare_adult_dataset)
    }

    # Defining the list of black-boxes
    blackbox_list = {
        # 'lr': LogisticRegression,
        'gt': GradientBoostingClassifier,
        # 'nn': MLPClassifier,

    }

    K_list = {
        'german': 200,
        'compas': 500,
        'adult': 2000
    }

    print('Local explanation experiment is running...')

    for dataset_kw in datsets_list:
        print('dataset=', dataset_kw)

        # Reading a data set
        dataset_name, prepare_dataset_fn = datsets_list[dataset_kw]
        dataset = prepare_dataset_fn(dataset_name, path_data)

        # Splitting the data set into train and test sets
        X, y = dataset['X'], dataset['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for blackbox_name in blackbox_list:
            print('blackbox=', blackbox_name)

            # Creating and training black-box
            BlackBoxConstructor = blackbox_list[blackbox_name]
            blackbox = BlackBoxConstructor(random_state=42)
            blackbox.fit(X_train, y_train)
            pred_train = blackbox.predict(X_train)
            pred_test = blackbox.predict(X_test)
            bb_accuracy = accuracy_score(y_test, pred_test)
            print('blackbox accuracy=', bb_accuracy)

            dfX2E = build_df2explain(blackbox, X_train, dataset).to_dict('records')

            # Creating/opening a csv file for storing results
            experiment_results = open(path_exp + 'local_explanation_results_%s_%s_%s.csv' %
                                      (dataset_kw, blackbox_name, 'K_' + str(K_list[dataset_kw])), 'a')

            # Extracting instance-level feature contributions
            # method = 'shapley_sampling_values' | 'tree_explainer' | 'tree_interpreter'
            contributions, extractor = ContributionExtraction(blackbox, X_train, method='shapley_sampling_values')

            # Finding anomaly instances in the train set
            anomaly_indices = np.where(pred_train != y_train)[0]
            X_anomaly = X_train[anomaly_indices]

            # Creating a KNN model for contribution values
            K = K_list[dataset_kw]
            cKNN = NearestNeighbors(n_neighbors=K).fit(contributions)

            # Selecting an instance to explain
            index = 0
            instance2explain = X_anomaly[index]
            contribution_x = extractor(instance2explain)
            _, nbrs_cKNN = cKNN.kneighbors(contribution_x.reshape(1, -1))
            nbrs_cKNN = nbrs_cKNN[0]

            # Picking representative samples
            B = 10
            N_top = 5
            contributions_nbrs = contributions[nbrs_cKNN]
            rp_ind = RepresentativePick(B, N_top, contributions_nbrs, nbrs_cKNN)
            rp_set = X_train[rp_ind]

            # Explaining isntance2explain using EXPLAN
            tau = 250
            N_samples = 3000
            exp_EXPLAN, info_EXPLAN = explan.Explainer(instance2explain,
                                                       blackbox,
                                                       dataset,
                                                       N_samples=N_samples,
                                                       tau=tau)

            # Printing the results
            print('\n')
            print('instance2explain =', str(dfX2E[index]))
            print('ground-truth =', str(y_train[anomaly_indices[index]]))
            print('blackbox-pred =', str(pred_train[anomaly_indices[index]]))
            print('explanation = %s' % exp_EXPLAN[1])
            print('\n')

            # Writing the results into the csv file
            results = '%s,%s\n%s,%s\n%s,%s\n%s,%s\n\n' % \
                      ('instance2explain =', str(dfX2E[index]),
                      'ground-truth =', str(y_train[anomaly_indices[index]]),
                      'blackbox-pred =', str(pred_train[anomaly_indices[index]]),
                      'explanation =', str(exp_EXPLAN[1]))
            experiment_results.write(results)

            # Explaining the representative set using EXPLAN
            tau = 250
            N_samples = 3000
            for b in range(B):
                exp_EXPLAN, info_EXPLAN = explan.Explainer(rp_set[b],
                                                           blackbox,
                                                           dataset,
                                                           N_samples=N_samples,
                                                           tau=tau)

                # Printing the results
                dfx = dfX2E[rp_ind[b]]
                print('representative %s = %s' % (b, dfx))
                print('ground-truth  = %s' % y_train[rp_ind[b]])
                print('blackbox-pred = %s' % pred_train[rp_ind[b]])
                print('explanation = %s' % exp_EXPLAN[1])
                print('\n')

                # Writing the results into the csv file
                results = '%s,%s\n%s,%s\n%s,%s\n%s,%s\n\n' % \
                          ('representaive ' + str(b) + ' =', str(dfx),
                          'ground-truth =', str(y_train[rp_ind[b]]),
                          'blackbox-pred =', str(pred_train[rp_ind[b]]),
                          'explanation =', str(exp_EXPLAN[1]))
                experiment_results.write(results)

            results = '\n'
            experiment_results.write(results)
            experiment_results.flush()
            experiment_results.close()

if __name__ == "__main__":
    main()
