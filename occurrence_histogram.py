from utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from contribution_extraction import ContributionExtraction
import matplotlib.pyplot as plt
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
        # 'nn': MLPClassifier
    }

    K_list = {
        'german': 200,
        'compas': 500,
        'adult': 2000
    }

    print('Occurrence histogram experiment is running...')

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
            pred_test = blackbox.predict(X_test)
            bb_accuracy = accuracy_score(y_test, pred_test)
            print('blackbox accuracy=', bb_accuracy)

            # Extracting instance-level feature contributions
            # method = 'shapley_sampling_values' | 'tree_explainer' | 'tree_interpreter'
            contributions, extractor = ContributionExtraction(blackbox, X_train, method='shapley_sampling_values')

            # Finding anomaly instances in the train set
            anomaly_indices = np.where(pred_train != y_train)[0]
            X_anomaly = X_train[anomaly_indices]

            # Creating KNN models for contribution values and feature values
            K = K_list[dataset_kw]
            cKNN = NearestNeighbors(n_neighbors=K).fit(contributions)
            fKNN = NearestNeighbors(n_neighbors=K).fit(X_train)

            # Finding occurrence histogram of training samples in the neighborhood of anomalies
            cHistogram = np.zeros(len(X_train))
            fHistogram = np.zeros(len(X_train))

            # cKNN
            contributions = extractor(X_anomaly)
            _, nbrs_cKNN = cKNN.kneighbors(contributions)
            for n in (nbrs_cKNN):
                cHistogram[n] = cHistogram[n] + 1

            # fKNN
            _, nbrs_fKNN = fKNN.kneighbors(X_anomaly)
            for n in (nbrs_fKNN):
                fHistogram[n] = fHistogram[n] + 1

            # Plot the occurrence histograms
            cSorted = np.argsort(cHistogram)
            fSorted = np.argsort(fHistogram)
            plt.plot(range(len(X_train)), cHistogram[cSorted], linewidth=2, color ='#be5683')
            plt.plot(range(len(X_train)), fHistogram[fSorted], linewidth=2, color ='#93b5e1')
            plt.xlabel('Training Samples')
            plt.ylabel('Number of Occurrence')
            data_name = str.upper(dataset_kw) if dataset_kw=='compas' else str.capitalize(dataset_kw)
            plt.title(data_name+ ' data set')
            plt.legend(['N_model_c', 'N_model_f'])
            plt.grid()
            plt.savefig(path_exp + 'occurrence_histogram_' + dataset_kw +
                        '_' + blackbox_name + '_' + 'K_' + str(K) + '.pdf')
            plt.show(block=False)
            plt.close()

if __name__ == "__main__":
    main()
