import numpy as np
import argparse
import os
import shutil
import time

from src import clf_utilities as clf_ut, writers as wrtrs
from src.config import config


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
    args = vars(ap.parse_args())

    if not clf_ut.is_valid(args['classifier']):
        return

    # TODO: add RandomizedSearch except for GridSearch
    params_grids = clf_ut.clf_hparams_map[args['classifier']][0]
    if isinstance(params_grids, list) is False:
        params_grids = [params_grids]

    t1 = time.time()
    results = []
    for i in range(1, config.n_folds + 1):
        fold_path = os.path.join(config.base_dir, args['experiment_path'], f'features_extraction_results/fold_{i}')
        X_train = np.load(os.path.join(fold_path, 'X_train.npy'))
        X_test = np.load(os.path.join(fold_path, 'X_test.npy'))
        y_train = np.load(os.path.join(fold_path, 'y_train.npy'))
        y_test = np.load(os.path.join(fold_path, 'y_test.npy'))
        for params_grid in params_grids:
            for params in clf_ut.create_clf_params_product_generator(params_grid):
                clf = clf_ut.clf_callable_map[args['classifier']].set_params(**params)
                clf.fit(X_train, y_train)
                pred = clf.predict_proba(X_test)
                y_pred = np.argsort(-pred, axis=1)[:, :]
                info = {'fold': i, 'clf_params': str(params)}
                scores = clf_ut.evaluate(y_test, y_pred)
                results.append(dict(info, **scores))

    results_path = os.path.join(config.base_dir, args['experiment_path'], 'model_selection_results')
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)

    wrtrs.write_clf_space(os.path.join(results_path, 'clf_space.csv'), args['classifier'])
    wrtrs.write_results(results_path, results, 'model_selection')

    print(f'Model selection done in {time.time() - t1:.3f} sec.')


if __name__ == "__main__":
    main()
