import numpy as np
import os
import shutil
import argparse
import pickle
import time

import clf_utilities as clf_ut
import writers as wrtrs
from config import config


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
    ap.add_argument('-feature_selection', required=False, action='store_true')
    ap.add_argument('-fs_method', required=False)
    args = vars(ap.parse_args())

    features_path = args['experiment_path'] + 'features_extraction_results'

    t1 = time.time()
    results = []
    for i in range(1, config.n_folds + 1):
        print('Fold:', i)
        fold_path = features_path + f'/fold_{i}'
        X_train_all = np.load(fold_path + '/X_train.npy')
        X_test_all = np.load(fold_path + '/X_test.npy')
        y_train = np.load(fold_path + '/y_train.npy')
        y_test = np.load(fold_path + '/y_test.npy')
        print('Number of features before feature selection {}'.format(X_train_all.shape[1]))
        config.total_features = X_train_all.shape[1]

        for clf_name in config.included_classifiers:
            print('Classifier:', clf_name)
            if clf_name == 'Baseline':
                encoder = pickle.load(open(features_path + '/encoder.pkl', 'rb'))
                baseline_service_enc = encoder.transform([config.baseline_service])[0]
                y_pred = np.tile(baseline_service_enc, (len(X_test_all), len(config.services)))
            else:

                if args['feature_selection']:
                    print('With feature selection {}'.format(args['fs_method']))
                    X_train, y_train, X_test, feature_ind = clf_ut.ft_selection(clf_name, args['fs_method'], X_train_all, y_train, X_test_all)
                else:
                    X_train, X_test = X_train_all, X_test_all
                    feature_ind = []

                print('Number of features {}'.format(X_train.shape[1]))
                clf = clf_ut.train_classifier(clf_name, X_train, y_train)
                pred = clf.predict_proba(X_test)
                y_pred = np.argsort(-pred, axis=1)[:, :]
            info = {'fold': i, 'classifier': clf_name, 'feature_col': list(feature_ind)}
            scores = clf_ut.evaluate(y_test, y_pred)
            results.append(dict(info, **scores))

    results_path = args['experiment_path'] + 'algorithm_selection_results'
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)

    wrtrs.write_results(results_path, results, 'algorithm_selection')

    print(f'Algorithm selection done in {time.time() - t1:.3f} sec.')
    return


if __name__ == '__main__':
    main()
