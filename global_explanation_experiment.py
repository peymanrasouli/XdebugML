from utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from contribution_extraction import ContributionExtraction
from alepython import ale_plot
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
        'compas': ('compas-scores-two-years.csv', prepare_compass_dataset),
        # 'adult': ('adult.csv', prepare_adult_dataset)
    }

    # Defining the list of black-boxes
    blackbox_list = {
        # 'lr': LogisticRegression,
        # 'gt': GradientBoostingClassifier,
        'nn': MLPClassifier,
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
            blackbox = BlackBoxConstructor(random_state=42)
            blackbox.fit(X_train, y_train)
            pred_train = blackbox.predict(X_train)
            pred_test = blackbox.predict(X_test)
            bb_accuracy = accuracy_score(y_test, pred_test)
            print('blackbox accuracy=', bb_accuracy)

            # Extracting instance-level feature contributions
            # method = 'shapley_sampling_values' | 'tree_interpreter'
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

            # Accumulated Local Effects (ALE) plots of neighborhood
            X_nbrs = X_train[nbrs_cKNN]

            features = dataset['columns'][1::]
            X_nbrs_df = pd.DataFrame(data=X_nbrs, columns=features)

            unique = X_nbrs_df.nunique().to_numpy()
            features = [features[f] for f in np.where(unique>1)[0]]

            for f in range(len(features)):
                ale_plot(blackbox, X_nbrs_df, features[f], monte_carlo=False)

if __name__ == "__main__":
    main()
