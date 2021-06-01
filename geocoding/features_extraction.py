import numpy as np
import os
import datetime
import argparse
import pickle
from shutil import copyfile
import time

from sklearn.model_selection import StratifiedKFold

from geocoding import features_utilities as feat_ut, writers as wrtrs
from geocoding.config import Config


def main():
    """
    Implements the first step of the experiment pipeline. Creates feature \
    sets for each one of the folds.

    Returns:
        None
    """
    # Construct argument parser and parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-fpath', required=True)
    args = vars(ap.parse_args())

    t1 = time.time()

    # Create folder to store experiment
    date_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    exp_path = os.path.join(Config.base_dir, 'experiments', f'exp_{date_time}')
    os.makedirs(exp_path)

    # Create folder to store feature extraction results
    results_path = os.path.join(exp_path, 'features_extraction_results')
    os.makedirs(results_path)
    copyfile('./geocoding/config.py', os.path.join(exp_path, 'config.py'))

    # Load dataset into dataframe
    df = feat_ut.load_points_df(os.path.join(Config.base_dir, args['fpath']))
    # Shuffle dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    # Encode labels
    df, encoder = feat_ut.encode_labels(df)

    df.to_csv(os.path.join(results_path, 'train_df.csv'), index=False)
    pickle.dump(encoder, open(os.path.join(results_path, 'encoder.pkl'), 'wb'))

    feat_ut.get_required_external_files(df, results_path)

    addresses, targets = list(df['address']), list(df['target'])
    skf = StratifiedKFold(n_splits=Config.n_folds)
    fold = 1

    for train_idxs, test_idxs in skf.split(addresses, targets):
        fold_path = os.path.join(results_path, 'fold_' + str(fold))
        os.makedirs(fold_path)
        os.makedirs(os.path.join(fold_path, 'features'))
        os.makedirs(os.path.join(fold_path, 'pickled_objects'))

        X_train = feat_ut.create_train_features(df.iloc[train_idxs].reset_index(), results_path, fold_path)
        X_test = feat_ut.create_test_features(
            df.iloc[test_idxs].reset_index(), results_path, os.path.join(fold_path, 'pickled_objects'), fold_path
        )
        y_train, y_test = df['target'][train_idxs], df['target'][test_idxs]

        np.save(os.path.join(fold_path, f'X_train.npy'), X_train)
        np.save(os.path.join(fold_path, f'X_test.npy'), X_test)
        np.save(os.path.join(fold_path, f'y_train.npy'), y_train)
        np.save(os.path.join(fold_path, f'y_test.npy'), y_test)
        fold += 1

    wrtrs.write_feats_space(os.path.join(results_path, 'included_features.csv'))

    print(f'Feature extraction done in {time.time() - t1:.3f} sec.')
    return


if __name__ == "__main__":
    main()
