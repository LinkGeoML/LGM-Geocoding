import pandas as pd
import csv

import clf_utilities as clf_ut
from config import config
import itertools as it
from collections import Counter
from operator import itemgetter


def write_feats_space(fpath):
    """
    Writes the features configuration in *fpath*.

    Args:
        fpath (str): Path to write

    Returns:
        None
    """
    with open(fpath, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['feature', 'normalized'])
        for f in config.included_features:
            writer.writerow([f, True if f in config.normalized_features else False])
    return


def write_clf_space(fpath, clf_name, best_params=None):
    """
    Writes *clf_name* classifier configuration in *fpath*. If *best_params* \
    is given then writes the best performing configuration of *clf_name*.

    Args:
        fpath (str): Path to write
        clf_name (str): Name of classifier to consider
        best_params (dict, optional): Has hyperparametrs as keys and the \
            corresponding values as values

    Returns:
        None
    """
    with open(fpath, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['classifier', 'parameters'])
        if best_params is None:
            writer.writerow([clf_name, clf_ut.clf_hparams_map[clf_name]])
        else:
            writer.writerow([clf_name, best_params])
    return


def write_results(results_path, results, step):
    """
    Writes full and averaged experiment results.

    Args:
        results_path (str): Path to write
        results (dict): Contains metrics as keys and the corresponding values \
            values
        step (str): Defines the experiment step

    Returns:
        None
    """
    col = 'classifier' if step == 'algorithm_selection' else 'clf_params'

    all_results_df = pd.DataFrame(results)
    all_results_df.to_csv(
        results_path + '/all_results.csv',
        columns=['fold', col, 'feature_col', 'accuracy',
                 'f1_macro', 'f1_micro', 'f1_weighted'],
        index=False)

    avg_results_df = pd.DataFrame(results)
    avg_results_df = avg_results_df.groupby(col).mean()
    if step == 'model_selection':
        avg_results_df = write_best_features_params(results, avg_results_df, step)
    avg_results_df.drop('fold', axis=1, inplace=True)
    avg_results_df.sort_values(by=['f1_weighted'], ascending=False, inplace=True)

    avg_results_df.to_csv(results_path + f'/results_by_{col}.csv', index=False)
    return


def write_predictions(fpath, df, preds):
    """
    Creates a csv file to present the predictions (in (predicted label, \
    score) pairs).

    Args:
        fpath (str): File path to write
        df (pandas.DataFrame): Contains the data points to which the \
            predictions refer to
        preds (list): Contains (predicted label, score) pairs

    Returns:
        None
    """
    n_services = len(config.services)
    with open(fpath, 'w', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['address', f'predictions'])
        for i in df.itertuples():
            writer.writerow([
                i.address,
                [
                    pred
                    for pred in preds[i.Index * n_services:i.Index * n_services + n_services]
                ]
            ])
    return


def write_best_features_params(results, avg_df, step):

    results_df = pd.DataFrame(results)
    grouped = results_df.groupby('clf_params')['feature_col'].apply(lambda x: list(it.chain(*x))).reset_index()
    min_features = results_df.groupby('clf_params')['feature_col'].apply(
        lambda x: min(map(len, x))).reset_index()
    min_features = dict(zip(min_features.clf_params, min_features.feature_col))
    grouped['feature_count'] = grouped['feature_col'].apply(lambda x: Counter(x))
    feat_fold = dict(zip(grouped.clf_params, grouped.feature_count))
    best_feat = {k: [i[0] for i in v.most_common(min_features[k])] for k, v in feat_fold.items()}
    best_feat_df = pd.DataFrame(data=list(best_feat.items()), columns=['clf_params', 'feature_col'])
    avg_df = avg_df.reset_index()
    avg_df = pd.merge(avg_df, best_feat_df, on='clf_params')

    return avg_df
