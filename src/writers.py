import pandas as pd
import csv
import os

from src import clf_utilities as clf_ut
from src.config import config


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
            writer.writerow([clf_name, clf_ut.clf_hparams_map[clf_name][0]])
        else:
            writer.writerow([clf_name, best_params])


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
        os.path.join(results_path, 'all_results.csv'),
        columns=['fold', col, 'accuracy', 'f1_macro', 'f1_micro', 'f1_weighted'],
        index=False
    )

    avg_results_df = all_results_df.groupby(col).mean()
    avg_results_df.drop('fold', axis=1, inplace=True)
    avg_results_df.sort_values(by=['accuracy'], ascending=False, inplace=True)
    avg_results_df.to_csv(os.path.join(results_path, f'results_by_{col}.csv'))


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
