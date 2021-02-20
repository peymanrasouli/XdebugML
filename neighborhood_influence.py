import os
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from contribution_extraction import ContributionExtraction
from influence_calculation import InfluenceCalculation
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
        'nn': MLPClassifier
    }

    K_list = {
        'german': 200,
        'compas': 500,
        'adult': 2000
    }

    print('Neighborhood influence experiment is running...')

    for dataset_kw in datsets_list:
        print('dataset=', dataset_kw)

        # Reading a data set
        dataset_name, prepare_dataset_fn = datsets_list[dataset_kw]
        dataset = prepare_dataset_fn(dataset_name, path_data)

        # Splitting the data set into train and test sets
        X, y = dataset['X'], dataset['y']
        X = MinMaxScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for blackbox_name in blackbox_list:
            print('blackbox=', blackbox_name)

            # Creating and training black-box
            BlackBoxConstructor = blackbox_list[blackbox_name]
            blackbox = BlackBoxConstructor(random_state=42)
            blackbox.fit(X_train, y_train)
            pred_train = blackbox.predict(X_train)
            pp_train = blackbox.predict_proba(X_train)
            pred_test = blackbox.predict(X_test)
            bb_accuracy = accuracy_score(y_test, pred_test)
            print('blackbox accuracy=', bb_accuracy)

            # Creating a csv file for storing results
            exists = os.path.isfile(path_exp + 'neighborhood_influence_results_%s_%s.csv' % (dataset_kw, blackbox_name))
            if exists:
                os.remove(path_exp + 'neighborhood_influence_results_%s_%s.csv' % (dataset_kw, blackbox_name))
            experiment_results = open(
                path_exp + 'neighborhood_influence_results_%s_%s.csv' % (dataset_kw, blackbox_name), 'a')

            results = '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
                      ('dataset', 'blackbox', 'bb_accuracy',
                       'iterations', 'K', 'perturb_percent',
                       'cKNN', 'fKNN', 'pKNN')
            experiment_results.write(results)

            results = '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
                      ('', '', '', '', '', '', '=average(G4:G1000)',
                       '=average(H4:H1000)', '=average(I4:I1000)')
            experiment_results.write(results)

            results = '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
                      ('', '', '', '', '', '', '=stdev(G4:G1000)',
                       '=stdev(H4:H1000)', '=stdev(I4:I1000)',)
            experiment_results.write(results)

            # Extracting instance-level feature contributions
            # method = 'shapley_sampling_values' | 'tree_explainer' | 'tree_interpreter'
            contributions, extractor = ContributionExtraction(blackbox, X_train, method='shapley_sampling_values')

            # Finding anomaly instances in the train set
            anomaly_indices = np.where(pred_train != y_train)[0]
            X_anomaly = X_train[anomaly_indices]

            # Creating KNN models for contribution values, feature values, and prediction probabilities
            K = K_list[dataset_kw]
            cKNN = NearestNeighbors(n_neighbors=K).fit(contributions)
            fKNN = NearestNeighbors(n_neighbors=K).fit(X_train)
            pKNN = NearestNeighbors(n_neighbors=K).fit(pp_train)

            # Main loop
            iter = 100
            n_test = 10
            for it in range(iter):
                print('Iteration=', it)
                perturb_percent = 1
                influence = InfluenceCalculation(blackbox, extractor, cKNN, fKNN, pKNN,
                                                 BlackBoxConstructor, X_train, y_train,
                                                 X_anomaly, n_test=n_test,
                                                 perturb_percent=perturb_percent)

                # Printing the results
                print('cKNN =', influence[0])
                print('fKNN =', influence[1])
                print('pKNN =', influence[2])
                print('\n')

                # Writing the results into the csv file
                results = '%s,%s,%.3f,%d,%d,%.2f,%.4f,%.4f,%.4f\n' % \
                          (dataset_kw, blackbox_name, bb_accuracy,
                           it, K, perturb_percent,
                           influence[0], influence[1], influence[2])
                experiment_results.write(results)
                experiment_results.flush()
            experiment_results.close()

if __name__ == "__main__":
    main()
