import pandas as pd
import argparse
import os
import shutil
import pickle
import time

from src import features_utilities as feat_ut, clf_utilities as clf_ut, writers as wrtrs
from src.config import config


def main():
    """
    Implements the fifth step of the experiment pipeline. This step loads a \
    pickled trained model from the previous step and deploys it in order to \
    make predictions on a test dataset.

    Returns:
        None
    """
    # Construct argument parser and parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-fpath', required=True)
    ap.add_argument('-experiment_path', required=True)
    args = vars(ap.parse_args())

    features_path = os.path.join(config.base_dir, args['experiment_path'], 'features_extraction_results')
    model_training_path = os.path.join(config.base_dir, args['experiment_path'], 'model_training_results')

    for path in [features_path, model_training_path]:
        if os.path.exists(path) is False:
            print('No such file:', path)
            return

    t1 = time.time()

    results_path = os.path.join(config.base_dir, 'experiments', args['experiment_path'], 'model_deployment_results')
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)
    os.makedirs(os.path.join(results_path, 'features'))

    df = feat_ut.load_points_df(os.path.join(config.base_dir, args['fpath']))
    encoder = pickle.load(open(os.path.join(features_path, 'encoder.pkl'), 'rb'))
    df, _ = feat_ut.encode_labels(df, encoder)

    features = list(pd.read_csv(os.path.join(features_path, 'included_features.csv'))['feature'])

    feat_ut.get_required_external_files(df, results_path, features)

    X_test = feat_ut.create_test_features(df, results_path, os.path.join(model_training_path, 'pickled_objects'), results_path, features)
    model = pickle.load(open(os.path.join(model_training_path, 'model.pkl'), 'rb'))
    preds = clf_ut.get_predictions(model, X_test)

    encoder = pickle.load(open(os.path.join(features_path, 'encoder.pkl'), 'rb'))
    preds = clf_ut.inverse_transform_labels(encoder, preds)

    wrtrs.write_predictions(os.path.join(results_path, 'predictions.csv'), df, preds)

    print(f'Model deployment done in {time.time() - t1:.3f} sec.')
    return


if __name__ == "__main__":
    main()
