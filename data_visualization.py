from EXPLAN.utils import *
from sklearn.ensemble import RandomForestClassifier
from treeinterpreter import treeinterpreter as ti
from sklearn.manifold import TSNE
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

# Reading a data set
dataset_name, prepare_dataset_fn = datsets_list['german']
dataset = prepare_dataset_fn(dataset_name, path_data)

X,y = dataset['X'], dataset['y']

blackbox = RandomForestClassifier(n_estimators=200)
blackbox.fit(X,y)
prediction, bias, contributions = ti.predict(blackbox, X)
contributions_ = np.zeros(np.shape(X))
for i in range(len(contributions_)):
    contributions_[i, :] = contributions[i, :, np.argmax(prediction[i])]

tsne_X =  TSNE(n_components=2)
X_2D = tsne_X.fit_transform(X)

tsne_C =  TSNE(n_components=2)
C_2D = tsne_C.fit_transform(contributions_)

color = ['tab:blue' if l==0 else 'tab:pink' for l in y]

plt.subplot(121)
plt.scatter(X_2D[:, 0], X_2D[:, 1], s=6, c=color)
plt.title("Feature Values")
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.scatter(C_2D[:, 0], C_2D[:, 1], s=6, c=color)
plt.title("Feature Contributions")
plt.xticks([])
plt.yticks([])
