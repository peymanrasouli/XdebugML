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
reduction_method = PCA    # TSNE | PCA

# Reading a data set
dataset_name, prepare_dataset_fn = datsets_list[data]
dataset = prepare_dataset_fn(dataset_name, path_data)
X,y = dataset['X'], dataset['y']

# Creating a black-box model
blackbox = RandomForestClassifier(random_state=42)
blackbox.fit(X,y)

# Extracting instance-level feature contributions
explanation_method = 'tree_explainer'   # 'shapley_sampling_values' | 'tree_explainer' | 'tree_interpreter'
contributions, extractor = ContributionExtraction(blackbox, X, method=explanation_method)

# Dimensionality reduction
X2D = reduction_method(n_components=2).fit_transform(X)
C2D = reduction_method(n_components=2).fit_transform(contributions)

# Plotting data points in feature value representation
plt.subplot(121)
X2D_0 = X2D[np.where(y==0)]
X2D_1 = X2D[np.where(y==1)]
class_0 = plt.scatter(X2D_0[:, 0], X2D_0[:, 1], s=3, c='#2e89ba', marker='o')
class_1 = plt.scatter(X2D_1[:, 0], X2D_1[:, 1], s=3, c='#f60c86', marker='o')
plt.title("Feature Values", fontsize=11)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xticks([])
plt.yticks([])
plt.legend((class_0, class_1),
           ('class 0', 'class 1'),
           scatterpoints=1,
           loc='upper right',
           ncol=1,
           fontsize=10)

# Plotting data points in feature contribution representation
plt.subplot(122)
C2D_0 = C2D[np.where(y==0)]
C2D_1 = C2D[np.where(y==1)]
class_0 = plt.scatter(C2D_0[:, 0], C2D_0[:, 1], s=3, c='#2e89ba', marker='o')
class_1 = plt.scatter(C2D_1[:, 0], C2D_1[:, 1], s=3, c='#f60c86', marker='o')
plt.title("Feature Contributions", fontsize=11)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xticks([])
plt.yticks([])
plt.legend((class_0, class_1),
           ('class 0', 'class 1'),
           scatterpoints=1,
           loc='upper right',
           ncol=1,
           fontsize=10)

plt.savefig(path_exp +'data_visualization_'+ explanation_method +'_' + dataset_name +'.pdf', bbox_inches='tight')
plt.show()