import numpy as np
from scipy.stats import randint as sp_randint, expon, truncnorm, uniform


class Config:

    """
    Class that configures the execution process.

    Attributes:
        n_folds (int): The number of folds in the experiment
        source_crs (int): The EPSG crs code used in input files
        target_crs (int): The EPSG crs code to transform the data
        clusters_pct (float): Percentage of data points, indicating how many \
            clusters to create in order to query Overpass API for streets
        osm_buffer (float): A buffer distance (in meters) to consider around \
            each bounding box when querying Overpass API
        osm_timeout (int): Timeout (in seconds) after five requests to \
            Overpass API
        max_overpass_tries (int): Maximum number of failed tries to extract the road network when querying the
            Overpass API before quiting.
        distance_thr (float): Distances in features greater than this value \
            will be converted to this threshold
        baseline_service (str): The name of the service to consider when \
            measuring baseline scores
        experiments_path (str): Path to folder that stores the experiments
        services (list): The services (geocoders) used in the setup

        supported_features (list): List of the supported features to choose \
            from
        included_features (list): List of the features to be included in the \
            experiment
        normalized_features (list): List of features to be normalized

        supported_classifiers (list): List of the supported classifiers to \
            choose from
        included_classifiers (list): List of the classifiers to be included \
            in the experiment

        NB_hparams (dict): Parameters search space for Naive Bayes classifier
        NN_hparams (dict): Parameters search space for Nearest Neighbors \
            classifier
        LR_hparams (dict): Parameters search space for Logistic Regression \
            classifier
        SVM_hparams (list): Parameters search space for SVM classifier
        MLP_hparams (dict): Parameters search space for MLP classifier
        DT_hparams (dict): Parameters search space for Decision Tree classifier
        RF_hparams (dict): Parameters search space for Random Forest classifier
        ET_hparams (dict): Parameters search space for Extra Trees classifier
    """

    n_folds = 5
    n_jobs = 4  #: int: Number of parallel jobs to be initiated. -1 means to utilize all available processors.
    # accepted values: randomized, grid, hyperband - not yet implemented!!!
    hyperparams_search_method = 'grid'
    """str: Search Method to use for finding best hyperparameters. (*randomized* | *grid*).
    """
    #: int: Number of iterations that RandomizedSearchCV should execute. It applies only when
    #: :attr:`hyperparams_search_method` equals to 'randomized'.
    max_iter = 30
    verbose = True

    source_crs = 4326
    target_crs = 3857
    clusters_pct = 0.015
    osm_buffer = 0.001
    osm_timeout = 50
    max_overpass_tries = 5
    distance_thr = 5000.0
    square_thr = 500000.0
    baseline_service = 'original'
    #: int: Seed to use by random number generators.
    seed_no = 13

    base_dir = '/media/disk/LGM-Geocoding'

    services = [
        'original',
        'arcgis',
        'nominatim',
    ]

    supported_features = [
        'normalized_coords',
        'pairwise_coords_distances',
        'pairwise_points_distances',
        'centroid_coords_distances',
        'centroid_points_distances',
        'mean_centroids_coords_distances',
        'mean_centroids_points_distances',
        'nearest_street_distance_per_service',
        'nearest_street_distance_by_centroid',
        'zip_codes',
        'common_nearest_street_distance',
        'intersects_on_common_nearest_street',
        'points_area',
        'polar_coords',
    ]

    included_features = [
        # 'normalized_coords',
        'pairwise_coords_distances',
        'pairwise_points_distances',
        'centroid_coords_distances',
        'centroid_points_distances',
        'mean_centroids_coords_distances',
        'mean_centroids_points_distances',
        'nearest_street_distance_per_service',
        'nearest_street_distance_by_centroid',
        # 'zip_codes',
        'common_nearest_street_distance',
        'intersects_on_common_nearest_street',
        'points_area',
        'polar_coords',
    ]

    normalized_features = [
        # 'normalized_coords',
        'pairwise_coords_distances',
        'pairwise_points_distances',
        'centroid_coords_distances',
        'centroid_points_distances',
        'mean_centroids_coords_distances',
        'mean_centroids_points_distances',
        'nearest_street_distance_per_service',
        'common_nearest_street_distance',
        'points_area',
        'polar_coords',
    ]

    supported_classifiers = [
        'Baseline',
        'NaiveBayes',
        'NearestNeighbors',
        'LogisticRegression',
        'SVM',
        'MLP',
        'DecisionTree',
        'RandomForest',
        'ExtraTrees',
        'XGBoost'
    ]

    included_classifiers = [
        'Baseline',
        'NaiveBayes',
        'NearestNeighbors',
        'LogisticRegression',
        'SVM',
        'MLP',
        'DecisionTree',
        'RandomForest',
        'ExtraTrees',
        'XGBoost'
    ]

    NB_hparams = {}

    NN_hparams = {
        'n_neighbors': [2, 3, 5, 10]
    }

    LR_hparams = {
        'max_iter': [100, 500],
        'C': [0.001, 0.1, 1, 10, 1000]
    }

    SVM_hparams = [
        {'kernel': ['rbf', 'sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
         'C': [0.001, 0.01, 0.1, 1, 10, 25, 50, 100, 1000], 'probability': [True]},
        {'kernel': ['poly'], 'degree': [1, 2, 3], 'gamma': ['scale', 'auto'],
         'C': [0.1, 1, 10, 25, 50, 100, 1000], 'max_iter': [10000], 'probability': [True]},
    ]

    MLP_hparams = {
        'hidden_layer_sizes': [(100, ), (50, 50, )],
        # 'learning_rate_init': [0.0001, 0.01, 0.1],
        'max_iter': [500, 1000],
        'solver': ['sgd', 'adam'],
    }

    DT_hparams = {
        'max_depth': [1, 4, 16, 32, 64],
        'min_samples_split': [2, 5, 10, 20, 50, 100, 200],
        'min_samples_leaf': [1, 2, 5, 10],
        'max_features': list(np.arange(2, 11, 2)) + ["sqrt", "log2", None]
    }

    RF_hparams = {
        'max_depth': [5, 10, 50, 100, 250, 300],
        'n_estimators': [100, 250, 500, 1000],
        # 'min_samples_leaf': [1, 5, 10],
        'min_samples_split': [2, 10],
    }

    ET_hparams = {
        'max_depth': [5, 10, 50, 100, 250, 300],
        'n_estimators': [100, 250, 500, 1000],
        # 'min_samples_leaf': [1, 5, 10],
        'min_samples_split': [2, 10, 50],
    }

    XGB_hparams = {
        "n_estimators": [500, 1000, 3000],
        'max_depth': [5, 10, 50, 100, 250, 300],
        # # hyperparameters to avoid overfitting
        # 'eta': list(np.linspace(0.01, 0.3, 10)),  # 'learning_rate'
        # 'gamma': [0, 1, 5],
        # 'subsample': [0.8, 0.9, 1],
        # # Values from 0.3 to 0.8 if you have many columns (especially if you did one-hot encoding),
        # # or 0.8 to 1 if you only have a few columns
        # 'colsample_bytree': list(np.linspace(0.8, 1, 3)),
        # 'min_child_weight': [1, 5, 10],
    }

    # These parameters constitute the search space for RandomizedSearchCV in our experiments.
    NB_hparams_dist = {}

    NN_hparams_dist = {
        'n_neighbors': sp_randint(1, 20)
    }

    LR_hparams_dist = {
        'max_iter': sp_randint(100, 1000),
        'C': expon(scale=200)
    }

    SVM_hparams_dist = {
        'C': expon(loc=0.01, scale=20),
        # "C": uniform(2, 10),
        "gamma": uniform(1e-5, 1e-2),
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'degree': sp_randint(1, 3),
        'class_weight': ['balanced', None],
        'tol': [1e-3, 1e-4],
        'max_iter': [100000],
        'probability': [True],
        # 'dual': [True, False]
    }
    DT_hparams_dist = {
        'max_depth': sp_randint(10, 200),
        'min_samples_split': sp_randint(2, 51),
        'min_samples_leaf': sp_randint(1, 15),
        # 'max_features': sp_randint(1, 11),
    }
    RF_hparams_dist = {
        'bootstrap': [True, False],
        # 'max_depth': [10, 20, 30, 40, 50, 60, 100, None],
        'max_depth': sp_randint(10, 300),
        "n_estimators": sp_randint(250, 2000),
        # 'criterion': ['gini', 'entropy'],
        # 'max_features': ['sqrt', 'log2'],  # sp_randint(1, 11)
        'min_samples_leaf': sp_randint(1, 10),
        'min_samples_split': sp_randint(2, 50),
    }
    XGB_hparams_dist = {
        "n_estimators": sp_randint(500, 4000),
        'max_depth': sp_randint(3, 300),
        # 'eta': expon(loc=0.01, scale=0.1),  # 'learning_rate'
        # hyperparameters to avoid overfitting
        'gamma': sp_randint(0, 5),
        'subsample': truncnorm(0.8, 1),
        'colsample_bytree': truncnorm(0.8, 1),
        'min_child_weight': sp_randint(1, 10),
    }
    MLP_hparams_dist = {
        'learning_rate_init': expon(loc=0.0001, scale=0.1),
        'max_iter': sp_randint(500, 2000),
        'solver': ['sgd', 'adam']
    }
