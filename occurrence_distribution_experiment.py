from EXPLAN.utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from treeinterpreter import treeinterpreter as ti
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
        'lr': LogisticRegression,
        'gt': GradientBoostingClassifier,
        'rf': RandomForestClassifier,
        'nn': MLPClassifier
    }

    K_list = {
        'german': 200,
        'compas': 500,
        'adult': 800
    }

    print('Occurrence distribution experiment is running...')

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

            # Random Forest surrogate model construction
            pred_train = blackbox.predict(X_train)
            pp_train = blackbox.predict_proba(X_train)
            surrogate = RandomForestClassifier(n_estimators=200)
            surrogate.fit(X_train, pred_train)
            prediction, bias, contributions = ti.predict(surrogate, X_train)
            contributions_ = np.zeros(np.shape(X_train))
            for i in range(len(contributions_)):
                contributions_[i,:] = contributions[i,:,np.argmax(prediction[i])]

            # Find anomaly instances in test set
            anomaly_indices = np.where(pred_test != y_test)[0]
            X_anomaly = X_test[anomaly_indices]

            # Creating KNN models for feature values and contribution values
            K = K_list[dataset_kw]
            cKNN = NearestNeighbors(n_neighbors=K).fit(contributions_)
            fKNN = NearestNeighbors(n_neighbors=K).fit(X_train)
            pKNN = NearestNeighbors(n_neighbors=K).fit(pp_train)

            # Finding occurrence distribution of training samples in the neighborhood of anomalies
            cDistribution = np.zeros(len(X_train))
            fDistribution = np.zeros(len(X_train))
            pDistribution = np.zeros(len(X_train))

            for i,x in zip(range(len(X_anomaly)),X_anomaly):
                print('anomaly sample=', i)

                prediction_x, bias_x, contribution_x = ti.predict(surrogate, x.reshape(1, -1))
                _, nbrs_cKNN = cKNN.kneighbors(contribution_x[:, :, np.argmax(prediction_x)].reshape(1, -1))
                nbrs_cKNN = nbrs_cKNN[0]
                cDistribution[nbrs_cKNN] = cDistribution[nbrs_cKNN] + 1

                _, nbrs_fKNN = fKNN.kneighbors(x.reshape(1, -1))
                nbrs_fKNN = nbrs_fKNN[0]
                fDistribution[nbrs_fKNN] = fDistribution[nbrs_fKNN] + 1

                _, nbrs_pKNN = pKNN.kneighbors(blackbox.predict_proba(x.reshape(1, -1)))
                nbrs_pKNN = nbrs_pKNN[0]
                pDistribution[nbrs_pKNN] = pDistribution[nbrs_pKNN] + 1

            # cDistribution bar plot
            cSorted = np.argsort(cDistribution)
            plt.bar(range(len(X_train)), cDistribution[cSorted])
            plt.xlabel('Training Samples')
            plt.ylabel('Number of Occurrence')
            plt.title('Occurrence distribution of training samples in the neighborhoods')
            plt.savefig(path_exp+'cKNN_'+dataset_kw+'_'+blackbox_name+'_'+'K_'+str(K)+'.pdf')
            plt.show(block=False)
            plt.close()

            # fDistribution bar plot
            fSorted = np.argsort(fDistribution)
            plt.bar(range(len(X_train)), fDistribution[fSorted])
            plt.xlabel('Training Samples')
            plt.ylabel('Number of Occurrence')
            plt.title('Occurrence distribution of training samples in the neighborhoods')
            plt.savefig(path_exp + 'fKNN_' + dataset_kw + '_' + blackbox_name + '_' + 'K_' + str(K) + '.pdf')
            plt.show(block=False)
            plt.close()

            # pDistribution bar plot
            pSorted = np.argsort(pDistribution)
            plt.bar(range(len(X_train)), pDistribution[pSorted])
            plt.xlabel('Training Samples')
            plt.ylabel('Number of Occurrence')
            plt.title('Occurrence distribution of training samples in the neighborhoods')
            plt.savefig(path_exp + 'pKNN_' + dataset_kw + '_' + blackbox_name + '_' + 'K_' + str(K) + '.pdf')
            plt.show(block=False)
            plt.close()

if __name__ == "__main__":
    main()
