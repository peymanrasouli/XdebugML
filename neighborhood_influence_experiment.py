import os
from EXPLAN.utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from treeinterpreter import treeinterpreter as ti
from neighborhood_influence import NeighborhoodInfluence
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for blackbox_name in blackbox_list:
            print('blackbox=', blackbox_name)

            # Creating and training black-box
            BlackBoxConstructor = blackbox_list[blackbox_name]
            blackbox = BlackBoxConstructor()
            blackbox.fit(X_train, y_train)
            pred_test = blackbox.predict(X_test)
            bb_accuracy = accuracy_score(y_test, pred_test)
            print('blackbox accuracy=', bb_accuracy)

            # Creating a csv file for storing results
            exists = os.path.isfile(path_exp + 'perturbation_influence_results_%s_%s.csv' % (dataset_kw, blackbox_name))
            if exists:
                os.remove(path_exp + 'perturbation_influence_results_%s_%s.csv' % (dataset_kw, blackbox_name))
            experiment_results = open(
                path_exp + 'perturbation_influence_results_%s_%s.csv' % (dataset_kw, blackbox_name), 'a')

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

            # Random Forest surrogate model construction
            pred_train = blackbox.predict(X_train)
            pp_train = blackbox.predict_proba(X_train)
            surrogate = RandomForestClassifier(n_estimators=200)
            surrogate.fit(X_train, pred_train)
            prediction, bias, contributions = ti.predict(surrogate, X_train)
            contributions_ = np.zeros(np.shape(X_train))
            for i in range(len(contributions_)):
                contributions_[i, :] = contributions[i, :, np.argmax(prediction[i])]

            # Find anomaly instances in test set
            anomaly_indices = np.where(pred_train != y_train)[0]
            X_anomaly = X_train[anomaly_indices]

            # Creating KNN models for feature values and contribution values
            K = K_list[dataset_kw]
            cKNN = NearestNeighbors(n_neighbors=K).fit(contributions_)
            fKNN = NearestNeighbors(n_neighbors=K).fit(X_train)
            pKNN = NearestNeighbors(n_neighbors=K).fit(pp_train)

            iter = 100
            n_test = 10
            for it in range(iter):
                print('Iteration=', it)
                perturb_percent = 0.5
                influence = NeighborhoodInfluence(blackbox, surrogate, cKNN, fKNN, pKNN, 
                				    BlackBoxConstructor, X_train, y_train,
                				    X_anomaly, n_test=n_test,
                                                  perturb_percent=perturb_percent)

                print('cKNN =', influence[0])
                print('fKNN =', influence[1])
                print('pKNN =', influence[2])
                print('\n')

                results = '%s,%s,%.3f,%d,%d,%.2f,%.4f,%.4f,%.4f\n' % \
                          (dataset_kw, blackbox_name, bb_accuracy,
                           it, K, perturb_percent,
                           influence[0], influence[1], influence[2])

                experiment_results.write(results)
                experiment_results.flush()
            experiment_results.close()

if __name__ == "__main__":
    main()
