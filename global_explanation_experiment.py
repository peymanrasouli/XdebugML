import os
import numpy as np
from EXPLAN.utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from treeinterpreter import treeinterpreter as ti
from alepython import ale_plot
from matplotlib import pyplot as plt
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
        # 'nn': MLPClassifier,
    }

    K_list = {
        'german': 200,
        'compas': 500,
        'adult': 2000
    }

    print('Global explanation experiment is running...')

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
            blackbox = BlackBoxConstructor()
            blackbox.fit(X_train, y_train)
            pred_test = blackbox.predict(X_test)
            bb_accuracy = accuracy_score(y_test, pred_test)
            print('blackbox accuracy=', bb_accuracy)

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

            # Selecting instance to explain
            index = 0
            instance2explain = X_anomaly[index]
            prediction_, bias_, contribution_ = ti.predict(surrogate, instance2explain.reshape(1, -1))
            _, nbrs_cKNN = cKNN.kneighbors(contribution_[:, :, np.argmax(prediction_)].reshape(1, -1))
            nbrs_cKNN = nbrs_cKNN[0]


            # Accumulated Local Effects (ALE) plots of neighborhood
            X_nbrs = X_train[nbrs_cKNN]

            features = dataset['columns'][1::]
            df_nbrs = pd.DataFrame(data=X_nbrs, columns=features)

            unique = df_nbrs.nunique().to_numpy()
            features = [features[f] for f in np.where(unique>1)[0]]

            for f in range(len(features)):
                ale_plot(blackbox,df_nbrs,features[f], monte_carlo=False)
                fig = plt.gcf()
                fig.savefig(path_exp+str(index)+'_'+features[f]+'_'+dataset_kw+'_'+blackbox_name+'.pdf')
                plt.close(fig)


if __name__ == "__main__":
    main()
