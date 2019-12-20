import numpy as np
import feature_selection as fs
from itertools import product
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
from sklearn.metrics import accuracy_score, f1_score

from config import config


clf_callable_map = {
    'NaiveBayes': GaussianNB(),
    'NearestNeighbors': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression(solver='liblinear', multi_class='auto'),
    'SVM': SVC(probability=True),
    'MLP': MLPClassifier(),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(random_state=1),
    'ExtraTrees': ExtraTreesClassifier()
}

clf_hparams_map = {
    'NaiveBayes': config.NB_hparams,
    'NearestNeighbors': config.NN_hparams,
    'LogisticRegression': config.LR_hparams,
    'SVM': config.SVM_hparams,
    'MLP': config.MLP_hparams,
    'DecisionTree': config.DT_hparams,
    'RandomForest': config.RF_hparams,
    'ExtraTrees': config.ET_hparams
}
"""
Feature selection mapping
"""

fs_callable_map = {
    'SelectKBest': [SelectKBest(chi2), config.SelectKbest_hyperparameters],
    'VarianceThreshold': [VarianceThreshold(1), config.VT_hyperparameters],
    'RFE': ['RFE', {}],
    'SelectFromModel': ['SelectFromModel', config.SFM_hyperparameters],
    'PCA': ['PCA']
}

feature_selection_getter_map = {
    'SelectKBest': ['get_stats_features', ('fs_name', 'fsm', 'clf', 'params', 'X_train', 'y_train', 'X_test')],
    'VarianceThreshold': ['get_stats_features', ('fs_name', 'fsm', 'clf', 'params', 'X_train', 'y_train', 'X_test')],
    'RFE': ['get_RFE_features', ('clf', 'clf_name', 'X_train', 'y_train', 'X_test')],
    'SelectFromModel': ['get_SFM_features', ('clf', 'clf_name', 'params', 'X_train', 'y_train', 'X_test')],
    'PCA': ['get_PCA_features', ('clf', 'params', 'X_train', 'y_train', 'X_test')],
}


def train_classifier(clf_name, X_train, y_train):
    """
    Trains a classifier through grid search.

    Args:
        clf_name (str): Classifier's name to be trained
        X_train (numpy.ndarray): Train features array
        y_train (numpy.ndarray): Train labels array

    Returns:
        object: The trained classifier
    """
    clf = clf_callable_map[clf_name]
    params = clf_hparams_map[clf_name]
    clf = GridSearchCV(clf, params, cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf


def evaluate(y_test, y_pred):
    """
    Evaluates model predictions through a series of metrics.

    Args:
        y_test (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels

    Returns:
        dict: Contains metrics names as keys and the corresponding values as \
        values
    """
    y_pred = y_pred[:, :1]
    scores = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_micro': f1_score(y_test, y_pred, average='micro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
    }
    return scores


def normalize_scores(scores):
    """
    Normalizes predictions scores to a probabilities-like format.

    Args:
        scores (list): Contains the predictions scores as predicted by the \
            model

    Returns:
        list: The normalized scores
    """
    s = sum(scores)
    normalized = [score/s for score in scores]
    return normalized


def get_predictions(model, X_test):
    """
    Makes predictions utilizing *model* over *X_test*.

    Args:
        model (object): The model to be used for predictions
        X_test (numpy.ndarray): The test features array

    Returns:
        list: Contains predictions in (label, score) pairs
    """
    preds = model.predict_proba(X_test)
    y_preds = []
    for pred in preds:
        labels = np.argsort(-pred)
        scores = normalize_scores(pred[labels])
        y_preds.append(zip(labels, scores))
    return y_preds


def inverse_transform_labels(encoder, preds):
    """
    Utilizes *encoder* to transform encoded labels back to the original \
    strings.

    Args:
        encoder (sklearn.preprocessing.LabelEncoder): The encoder to be \
            utilized
        k_preds (list): Contains predictions in (label, score) pairs

    Returns:
        list: Contains predictions in (label, score) pairs, where label is \
            now in the original string format
    """
    label_mapping = dict(
        zip(encoder.transform(encoder.classes_), encoder.classes_))
    k_preds_new = [(label_mapping[pred[0]], pred[1]) for k_pred in preds
                   for pred in k_pred]
    return k_preds_new


def is_valid(clf_name):
    """
    Checks whether *clf_name* is a valid classifier's name with respect to \
    the experiment setup.

    Args:
        clf_name (str): Classifier's name

    Returns:
        bool: Returns True if given classifier's name is valid
    """
    supported_clfs = [
        clf for clf in config.supported_classifiers if clf != 'Baseline'
    ]
    if clf_name not in supported_clfs:
        print('Supported classifiers:', supported_clfs)
        return False
    return True


def create_clf_params_product_generator(params_grid):
    """
    Generates all possible combinations of classifier's hyperparameters values.

    Args:
        params_grid (dict): Contains classifier's hyperparameters names as \
            keys and the correspoding search space as values

    Yields:
        dict: Contains a classifier's hyperparameters configuration
    """
    keys = params_grid.keys()
    vals = params_grid.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))


def ft_selection(clf_name, fs_method, X_train, y_train, X_test):
    clf = clf_callable_map[clf_name]
    fsm = fs_callable_map[fs_method][0]
    params = fs_callable_map[fs_method][1]
    args = create_ft_select_args_dict(fs_method, fsm, clf, clf_name, params, X_train, y_train, X_test)
    X_train, y_train, X_test, feature_indices = getattr(fs, feature_selection_getter_map[fs_method][0])(
        *[args[arg] for arg in feature_selection_getter_map[fs_method][1]]
    )
    return X_train, y_train, X_test, feature_indices


def create_ft_select_args_dict(fs_name, fsm, clf, clf_name, params, X_train, y_train, X_test):

    return locals()