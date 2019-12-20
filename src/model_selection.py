import numpy as np
import argparse
import os
import shutil
import time

import clf_utilities as clf_ut
import writers as wrtrs
from config import config


def main():
    """
    Implements the third step of the experiment pipeline. Given a classifier, \
    this step is responsible to find the best performing classifier \
    configuration.

    Returns:
        None
    """
    # Construct argument parser and parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-classifier', required=True)
    ap.add_argument('-experiment_path', required=True)
    ap.add_argument('-feature_selection', required=False, action='store_true')
    ap.add_argument('-fs_method', required=False)

    args = vars(ap.parse_args())

    if not clf_ut.is_valid(args['classifier']):
        return

    params_grids = clf_ut.clf_hparams_map[args['classifier']]
    if isinstance(params_grids, list) is False:
        params_grids = [params_grids]

    t1 = time.time()
    print("Feature selection with {}".format(args['fs_method']))

    results = []
    for i in range(1, config.n_folds + 1):
        fold_path = args['experiment_path'] + f'features_extraction_results/fold_{i}'
        X_train_all = np.load(fold_path + '/X_train_all.npy')

        X_test_all = np.load(fold_path + '/X_test_all.npy')
        print("# of features before selection {}".format(X_train_all.shape[1]))
        y_train = np.load(fold_path + '/y_train.npy')
        y_test = np.load(fold_path + '/y_test.npy')
        if args['feature_selection']:
            X_train, y_train, X_test, feature_ind = clf_ut.ft_selection(args['classifier'], args['fs_method'], X_train_all, y_train,
                                                           X_test_all)
        else:
            X_train, X_test = X_train_all, X_test_all
            print("Features selected {}".format(X_train.shape[1]))

        for params_grid in params_grids:
            for params in clf_ut.create_clf_params_product_generator(params_grid):
                clf = clf_ut.clf_callable_map[args['classifier']].set_params(**params)
                clf.fit(X_train, y_train)
                pred = clf.predict_proba(X_test)
                y_pred = np.argsort(-pred, axis=1)[:, :]
                info = {'fold': i, 'clf_params': str(params), 'feature_col': list(feature_ind)}
                scores = clf_ut.evaluate(y_test, y_pred)
                results.append(dict(info, **scores))

    results_path = args['experiment_path'] + 'model_selection_results'
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)

    wrtrs.write_clf_space(results_path + '/clf_space.csv', args['classifier'])
    wrtrs.write_results(results_path, results, 'model_selection')

    print(f'Model selection done in {time.time() - t1:.3f} sec.')
    return


if __name__ == "__main__":
    main()
