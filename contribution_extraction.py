from utils import *
from shap import SamplingExplainer, TreeExplainer
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def ContributionExtraction(blackbox, X_train, method='shapley_sampling_values'):

    ## Shapley sampling values method
    if method is 'shapley_sampling_values':
        pred_train = blackbox.predict(X_train)
        explainer = SamplingExplainer(blackbox.predict_proba, X_train)
        contributions_ = explainer.shap_values(X_train, nsamples=1000)
        contributions = np.zeros(np.shape(X_train))
        for i in range(len(contributions)):
            contributions[i, :] = contributions_[pred_train[i]][i, :]

        def extractor(X):
            if len(X.shape)==1:
                l_x = blackbox.predict(X.reshape(1, -1))[0]
                contribution_x = explainer.shap_values(X.reshape(1, -1), nsamples=1000)
                return contribution_x[l_x]
            else:
                l_X = blackbox.predict(X)
                contributions_X_ = explainer.shap_values(X, nsamples=1000)
                contributions_X = np.zeros(np.shape(X))
                for i in range(len(contributions_X)):
                    contributions_X[i, :] = contributions_X_[l_X[i]][i, :]
                return contributions_X

        return contributions, extractor

    ## TreeExplainer method
    elif method is 'tree_explainer':
        pred_train = blackbox.predict(X_train)
        surrogate = XGBClassifier(n_estimators=200)
        surrogate.fit(X_train, pred_train)
        explainer = TreeExplainer(surrogate)
        contributions = explainer.shap_values(X_train)

        def extractor(X):
            if len(X.shape) == 1:
                contribution_x = explainer.shap_values(X.reshape(1, -1))
                return contribution_x
            else:
                contributions_X = explainer.shap_values(X)
                return contributions_X

        return contributions, extractor

    ## TreeInterpreter method
    elif method is 'tree_interpreter':
        pred_train = blackbox.predict(X_train)
        surrogate = RandomForestClassifier(n_estimators=200)
        surrogate.fit(X_train, pred_train)
        prediction, bias, contributions_ = treeinterpreter.predict(surrogate, X_train)
        contributions = np.zeros(np.shape(X_train))
        for i in range(len(contributions)):
            contributions[i, :] = contributions_[i, :, np.argmax(prediction[i])]

        def extractor(X):
            if len(X.shape) == 1:
                prediction_x, bias_x, contribution_x = treeinterpreter.predict(surrogate, X.reshape(1, -1))
                l_x = np.argmax(prediction_x)
                return contribution_x[:,:,l_x]
            else:
                prediction_X, bias_X, contributions_X_ = treeinterpreter.predict(surrogate, X)
                l_X = np.argmax(prediction_X,axis=1)
                contributions_X = np.zeros(np.shape(X))
                for i in range(len(contributions_X)):
                    contributions_X[i, :] = contributions_X_[i, :, l_X[i]]
                return contributions_X

        return contributions, extractor
