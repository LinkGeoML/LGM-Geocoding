import pandas as pd
import argparse
from ast import literal_eval
import os
import shutil
import pickle
import time

import features_utilities as feat_ut
import clf_utilities as clf_ut
import writers as wrtrs


def main():
    """
    Implements the fourth step of the experiment pipeline. This step reads \
    the best performing configuration from the previous steps and trains the \
    corresponding model on all available train dataset.

    Returns:
        None
    """
    # Construct argument parser and parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-experiment_path', required=True)
    args = vars(ap.parse_args())

    features_path = args['experiment_path'] + 'features_extraction_results'
    model_selection_path = args['experiment_path'] + 'model_selection_results'

    for path in [model_selection_path, features_path]:
        if os.path.exists(path) is False:
            print('No such file:', path)
            return

    t1 = time.time()

    results_path = args['experiment_path'] + 'model_training_results'
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)
    os.makedirs(results_path + '/features')
    os.makedirs(results_path + '/pickled_objects')

    df = pd.read_csv(features_path + '/train_df.csv')
    path = model_selection_path + '/clf_space.csv'
    clf_name = pd.read_csv(path, nrows=1).iloc[0, 0]
    path = model_selection_path + '/results_by_clf_params.csv'
    params = literal_eval(pd.read_csv(path, nrows=1).iloc[0, 0])
    features_selected = literal_eval(pd.read_csv(path, nrows=1).iloc[0,5])

    features = list(pd.read_csv(features_path + '/included_features.csv')['feature'])
    X_train = feat_ut.create_train_features(df, features_path, results_path, features)
    if features_selected:
        X_train = X_train[:, features_selected]
    y_train = df['target']

    model = clf_ut.clf_callable_map[clf_name].set_params(**params)
    model.fit(X_train, y_train)

    pickle.dump(model, open(results_path + '/model.pkl', 'wb'))

    wrtrs.write_clf_space(results_path + '/model_config.csv', clf_name, params)

    print(f'Model training done in {time.time() - t1:.3f} sec.')
    return


if __name__ == "__main__":
    main()
