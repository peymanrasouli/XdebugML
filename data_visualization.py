from utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from contribution_extraction import ContributionExtraction
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

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

# Selecting the data and the dimensionality reduction method
data = 'german'  # 'german' | 'compas' | 'adult'
method = TSNE    # TSNE | PCA

# Reading a data set
dataset_name, prepare_dataset_fn = datsets_list[data]
dataset = prepare_dataset_fn(dataset_name, path_data)
X,y = dataset['X'], dataset['y']

# Creating a black-box model
blackbox = RandomForestClassifier(random_state=42)
blackbox.fit(X,y)

# Extracting instance-level feature contributions
# method = 'shapley_sampling_values' | 'tree_explainer' | 'tree_interpreter'
contributions, extractor = ContributionExtraction(blackbox, X, method='tree_explainer')

# Dimensionality reduction
X2D = method(n_components=2).fit_transform(X)
C2D = method(n_components=2).fit_transform(contributions)

# Plotting the distribution of the data in both spaces
color = ['#2e89ba' if l==0 else '#f60c86' for l in y]

plt.subplot(121)
plt.scatter(X2D[:, 0], X2D[:, 1], s=3, c=color, marker='o')
plt.title("Feature Values")
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.scatter(C2D[:, 0], C2D[:, 1], s=3, c=color, marker='o')
plt.title("Feature Contributions")
plt.xticks([])
plt.yticks([])
