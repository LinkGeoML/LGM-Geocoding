import numpy as np
import os
import shutil
import argparse
import pickle
import time
import warnings

from src import clf_utilities as clf_ut, writers as wrtrs
from src.config import config


warnings.filterwarnings('ignore', 'Solver terminated early.*')


def main():
    """
    Implements the second step of the experiment pipeline. Trains a series of \
    classifiers based on different configurations in a nested cross \
    validation scheme.

    Returns:
        None
    """
    # Construct argument parser and parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-experiment_path', required=True)
    args = vars(ap.parse_args())

    features_path = os.path.join(config.base_dir, args['experiment_path'], 'features_extraction_results')

    t1 = time.time()
    results = []
    for i in range(1, config.n_folds + 1):
        print('Fold:', i)
        fold_path = features_path + f'/fold_{i}'
        X_train = np.load(fold_path + '/X_train.npy')
        X_test = np.load(fold_path + '/X_test.npy')
        y_train = np.load(fold_path + '/y_train.npy')
        y_test = np.load(fold_path + '/y_test.npy')
        for clf_name in config.included_classifiers:
            print('Classifier:', clf_name)
            params = {'hparams': None}
            if clf_name == 'Baseline':
                encoder = pickle.load(open(os.path.join(features_path, 'encoder.pkl'), 'rb'))
                baseline_service_enc = encoder.transform([config.baseline_service])[0]
                y_pred = np.tile(baseline_service_enc, (len(X_test), len(config.services)))
            else:
                clf = clf_ut.train_classifier(clf_name, X_train, y_train)
                params['hparams'] = clf.best_params_
                pred = clf.predict_proba(X_test)
                y_pred = np.argsort(-pred, axis=1)[:, :]
            info = {'fold': i, 'classifier': clf_name}
            scores = clf_ut.evaluate(y_test, y_pred)
            # results.append(dict(info, **scores, **params))
            results.append(dict(info, **scores))

    results_path = os.path.join(config.base_dir, args['experiment_path'], 'algorithm_selection_results')
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)

    wrtrs.write_results(results_path, results, 'algorithm_selection')

    print(f'Algorithm selection done in {time.time() - t1:.3f} sec.')


if __name__ == '__main__':
    main()
