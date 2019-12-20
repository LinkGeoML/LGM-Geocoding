class config:

    """
    Class that defines the experiment configuration.

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
    source_crs = 4326
    target_crs = 3857
    clusters_pct = 0.015
    osm_buffer = 0.001
    osm_timeout = 150
    distance_thr = 5000.0
    baseline_service = 'original'
    total_features = 0
    experiments_path = 'media/disk/LGM-Geocoding-utils/experiments'
    fs_method = ''
    feature_selection = False

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
        'zip_codes'
    ]

    included_features = [
        'normalized_coords',
        'pairwise_coords_distances',
        'pairwise_points_distances',
        'centroid_coords_distances',
        'centroid_points_distances',
        'mean_centroids_coords_distances',
        'mean_centroids_points_distances',
        'nearest_street_distance_per_service',
        'nearest_street_distance_by_centroid',
        'zip_codes'
    ]

    normalized_features = [
        'normalized_coords',
        'pairwise_coords_distances',
        'pairwise_points_distances',
        'centroid_coords_distances',
        'centroid_points_distances',
        'mean_centroids_coords_distances',
        'mean_centroids_points_distances',
        'nearest_street_distance_per_service'
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
        'ExtraTrees'
    ]

    included_classifiers = [
        #'Baseline',
        #'NaiveBayes',
        #'NearestNeighbors',
        #'LogisticRegression',
        # 'SVM',
        #'MLP',
        #'DecisionTree',
        'RandomForest',
        #'ExtraTrees'
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
        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [0.01, 0.1, 1, 10, 100]},
        {'kernel': ['poly'], 'degree': [1, 2, 3], 'C': [0.01, 0.1, 1, 10, 100]},
    ]

    MLP_hparams = {
        'hidden_layer_sizes': [(100, ), (50, 50, )],
        'learning_rate_init': [0.0001, 0.01, 0.1],
        'max_iter': [100, 200, 500]
    }

    DT_hparams = {
        'max_depth': [1, 4, 16, 32],
        'min_samples_split': [0.1, 0.2, 0.5, 1.0]
    }

    RF_hparams = {
        'max_depth': [5, 10, 100, 250, None],
        'n_estimators': [100, 250, 1000]
    }

    ET_hparams = {
        'max_depth': [5, 10, 100, 250],
        'n_estimators': [100, 250, 1000]
    }

    """
       Feature selection hyperparameters
    """

    SelectKbest_hyperparameters = {
        'selection__k': [.9, .8, .7]
    }

    VT_hyperparameters = [
        {
            'selection__threshold': [0.001, 0.0008, 0.002]
        }
    ]

    SFM_hyperparameters = [
        {
            'selection__threshold': ['0.3*median', '0.5*median', '0.7*mean']
        }
    ]

    PCA_hyperparameters = [
        {
            'selection__n_components': [.9, .8, .7]
        }
    ]