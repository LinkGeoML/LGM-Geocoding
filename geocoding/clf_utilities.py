import numpy as np
from itertools import product
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
# from xgboost import XGBClassifier

from geocoding.config import Config


clf_callable_map = {
    'NaiveBayes': GaussianNB(),
    'NearestNeighbors': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression(solver='liblinear', multi_class='auto'),
    'SVM': SVC(),
    'MLP': MLPClassifier(),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'ExtraTrees': ExtraTreesClassifier(),
    # 'XGBoost': XGBClassifier()
}

clf_hparams_map = {
    'NaiveBayes': [Config.NB_hparams, Config.NB_hparams_dist],
    'NearestNeighbors': [Config.NN_hparams, Config.NN_hparams_dist],
    'LogisticRegression': [Config.LR_hparams, Config.LR_hparams_dist],
    'SVM': [Config.SVM_hparams, Config.SVM_hparams_dist],
    'MLP': [Config.MLP_hparams, Config.MLP_hparams_dist],
    'DecisionTree': [Config.DT_hparams, Config.DT_hparams_dist],
    'RandomForest': [Config.RF_hparams, Config.RF_hparams_dist],
    'ExtraTrees': [Config.ET_hparams, Config.RF_hparams_dist],
    'XGBoost': [Config.XGB_hparams, Config.XGB_hparams_dist],
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
    if Config.hyperparams_search_method.lower() == 'grid':
        params = clf_hparams_map[clf_name][0]
        clf = GridSearchCV(clf, params, cv=Config.n_folds, n_jobs=Config.n_jobs, verbose=Config.verbose)
        # elif self.search_method.lower() == 'hyperband' and clf_key in ['XGBoost', 'Extra-Trees', 'Random Forest']:
        #     HyperbandSearchCV(
        #         clf_val[0](probability=True) if clf_key == 'SVM' else clf_val[0](), clf_val[2].copy().pop('n_estimators'),
        #         resource_param='n_estimators',
        #         min_iter=500 if clf_key == 'XGBoost' else 200,
        #         max_iter=3000 if clf_key == 'XGBoost' else 1000,
        #         cv=self.inner_cv, random_state=seed_no, scoring=score
        #     )
    else:  # randomized is used as default
        params = clf_hparams_map[clf_name][1]
        clf = RandomizedSearchCV(
            clf, params, cv=Config.n_folds, n_jobs=Config.n_jobs, verbose=Config.verbose, n_iter=Config.max_iter
        )
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
        clf for clf in Config.supported_classifiers if clf != 'Baseline'
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
