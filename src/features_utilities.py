import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.wkt import loads
import pickle
import os

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler

from src import features as feats, osm_utilities as osm_ut
from src.config import config


features_getter_map = {
    'normalized_coords': 'get_normalized_coords',
    'pairwise_coords_distances': 'get_pairwise_coords_distances',
    'pairwise_points_distances': 'get_pairwise_points_distances',
    'centroid_coords_distances': 'get_centroid_coords_distances',
    'centroid_points_distances': 'get_centroid_points_distances',
    'mean_centroids_coords_distances': 'get_mean_centroids_coords_distances',
    'mean_centroids_points_distances': 'get_mean_centroids_points_distances',
    'nearest_street_distance_per_service': 'get_nearest_street_distance_per_service',
    'nearest_street_distance_by_centroid': 'get_nearest_street_distance_by_centroid',
    'zip_codes': 'get_zip_codes',
    'common_nearest_street_distance': 'get_common_nearest_street_distance',
    'intersects_on_common_nearest_street': 'get_intersects_on_common_nearest_street',
    'points_area': 'get_points_area',
    'polar_coords': 'get_polar_coords',
}

features_getter_args_map = {
    'normalized_coords': ['df'],
    'pairwise_coords_distances': ['df'],
    'pairwise_points_distances': ['df'],
    'centroid_coords_distances': ['df'],
    'centroid_points_distances': ['df'],
    'mean_centroids_coords_distances': ['df'],
    'mean_centroids_points_distances': ['df'],
    'nearest_street_distance_per_service': ['df', 'street_gdf'],
    'nearest_street_distance_by_centroid': ['df', 'street_gdf'],
    'zip_codes': ['df'],
    'common_nearest_street_distance': ['df', 'street_gdf'],
    'intersects_on_common_nearest_street': ['df', 'street_gdf'],
    'points_area': ['df'],
    'polar_coords': ['df'],
}


def load_points_df(points_fpath):
    """
    Loads points in *points_fpath* into a pandas.DataFrame and project their \
    geometries.

    Args:
        points_fpath (str): Path to file containing the points

    Returns:
        pandas.DataFrame
    """
    df = pd.read_csv(points_fpath)
    for service in config.services:
        # service_df = df[[f'x_{service}', f'y_{service}']]
        # service_df['geometry'] = service_df.apply(lambda x: Point(x[f'x_{service}'], x[f'y_{service}']), axis=1)
        service_gdf = gpd.GeoDataFrame(
            df[[f'x_{service}', f'y_{service}']],
            geometry=gpd.points_from_xy(df[f'x_{service}'], df[f'y_{service}']),
            crs=f'epsg:{config.source_crs}'
        )
        # service_gdf = gpd.GeoDataFrame(service_df, geometry='geometry', crs=f'epsg:{config.source_crs}')
        # print(service_df.geometry.crs, f'epsg:{config.source_crs}')
        # service_gdf.crs = f'epsg:{config.source_crs}'
        service_gdf = service_gdf.to_crs(f'epsg:{config.target_crs}')
        df[f'lon_{service}'] = service_gdf.apply(lambda x: x.geometry.x, axis=1)
        df[f'lat_{service}'] = service_gdf.apply(lambda x: x.geometry.y, axis=1)
    return df


def encode_labels(df, encoder=None):
    """
    Encodes target column to with integer values.

    Args:
        df (pandas.DataFrame): The DataFrame containing the column to be \
            encoded
        encoder (sklearn.preprocessing.LabelEncoder, optional): The label \
            encoder to be utilized

    Returns:
        tuple:
            pandas.DataFrame: The DataFrame with the encoded column

            sklearn.preprocessing.LabelEncoder: The label encoder utilized
    """
    if encoder is None:
        encoder = LabelEncoder()
        df['target'] = encoder.fit_transform(df['label'])
    else:
        df['target'] = encoder.transform(df['label'])
    return df, encoder


def load_street_gdf(street_fpath):
    """
    Loads streets in *street_fpath* into a geopandas.GeoDataFrame and project \
    their geometries.

    Args:
        street_fpath (str): Path to file containing the streets

    Returns:
        geopandas.GeoDataFrame
    """
    street_df = pd.read_csv(street_fpath)
    street_df['geometry'] = street_df['geometry'].apply(lambda x: loads(x))
    street_gdf = gpd.GeoDataFrame(street_df, geometry='geometry', crs=f'epsg:{config.source_crs}')
    # street_gdf.crs = f'epsg:{config.source_crs}'
    street_gdf = street_gdf.to_crs(f'epsg:{config.target_crs}')
    return street_gdf


def prepare_feats_args(df, required_args, path):
    """
    Prepares required arguments during features extraction.

    Args:
        df (pandas.DataFrame): Contains the points for which features will be \
            created
        required_args (set): Contains the names of the required args
        path (str): Path to read from

    Returns:
        dict: Containing arguments names as keys and their corresponding \
            structures as values
    """
    args = {'df': df}
    if 'street_gdf' in required_args:
        args['street_gdf'] = load_street_gdf(os.path.join(path, 'osm_streets.csv'))
    return args


def create_train_features(df, in_path, out_path, features=None):
    """
    Creates all the included train features arrays and saves them in \
        *out_path*.

    Args:
        df (pandas.DataFrame): Contains the train points
        in_path (str): Path to read required items
        out_path (str): Path to write
        features (list, optional): Contains the names of the features to \
            extract

    Returns:
        numpy.ndarray: The train features array
    """
    included_features = config.included_features if features is None else features

    required_args = set([
        arg for f in included_features
        for arg in features_getter_args_map[f]
    ])
    args = prepare_feats_args(df, required_args, in_path)
    Xs = []
    for f in included_features:
        X = getattr(feats, features_getter_map[f])(*[args[arg] for arg in features_getter_args_map[f]])
        if f in config.normalized_features:
            X, scaler = normalize_features(X)
            pickle.dump(scaler, open(os.path.join(out_path, 'pickled_objects', f'{f}_scaler.pkl'), 'wb'))
        np.save(out_path + f'/features/{f}_train.npy', X)
        Xs.append(X)
    X = np.hstack(Xs)
    return X


def create_test_features(df, in_path, scalers_path, out_path, features=None):
    """
    Creates all the included test features arrays and saves them in \
        *out_path*.

    Args:
        df (pandas.DataFrame): Contains the test points
        in_path (str): Path to read required items
        scalers_path (str): Path to load required scalers
        out_path (str): Path to write
        features (list, optional): Contains the names of the features to \
            extract

    Returns:
        numpy.ndarray: The test features array
    """
    included_features = config.included_features if features is None else features

    required_args = set([
        arg for f in included_features
        for arg in features_getter_args_map[f]
    ])
    args = prepare_feats_args(df, required_args, in_path)
    Xs = []
    for f in included_features:
        X = getattr(feats, features_getter_map[f])(
            *[args[arg] for arg in features_getter_args_map[f]])
        if f in config.normalized_features:
            scaler = pickle.load(open(os.path.join(scalers_path, f'{f}_scaler.pkl'), 'rb'))
            X, _ = normalize_features(X, scaler)
        np.save(os.path.join(out_path, f'features/{f}_test.npy'), X)
        Xs.append(X)
    X = np.hstack(Xs)
    return X


def normalize_features(X, scaler=None):
    """
    Normalize features to [0, 1].

    Args:
        X (numpy.ndarray): Features array to be normalized
        scaler (sklearn.preprocessing.MinMaxScaler, optional): Scaler to be \
            utilized

    Returns:
        tuple:
            numpy.ndarray: The normalized features array

            sklearn.preprocessing.MinMaxScaler: The scaler utilized
    """
    if scaler is None:
        scaler = MinMaxScaler()
        # scaler = RobustScaler()
        X_ = scaler.fit_transform(X)
    else:
        X_ = scaler.transform(X)
    return X_, scaler


def filter(values):
    """
    Filters *values* by replacing values greater than *config.distance_thr* \
    with *config.distance_thr*.

    Args:
        values (list): Contains distances created by various features

    Returns:
        list: Contains the filtered distances
    """
    values_ = [
        config.distance_thr if v > config.distance_thr else round(v, 2)
        for v in values
    ]
    return values_


# def get_bbox_coords(df):
#     x_cols = [f'x_{service}' for service in config.services]
#     y_cols = [f'y_{service}' for service in config.services]
#     west, east = np.min(df[x_cols].values), np.max(df[x_cols].values)
#     south, north = np.min(df[y_cols].values), np.max(df[y_cols].values)
#     return (south, west, north, east)


# def get_centroids(df):
#     lons = df.apply(lambda x: np.mean([x[f'x_{service}'] for service in config.services]), axis=1)
#     lats = df.apply(lambda x: np.mean([x[f'y_{service}'] for service in config.services]), axis=1)
#     centroids = np.array(list(zip(lons, lats)))
#     return centroids


def get_points(df):
    """
    Builds an array of all points appearing in *df*. This array will have a \
    shape of (len(df) * number_of_services, 2).

    Args:
        df (pandas.DataFrame): Contains the data points

    Returns:
        numpy.ndarray
    """
    points = [df[[f'x_{service}', f'y_{service}']].to_numpy() for service in config.services]
    points = np.vstack(points)

    return points


def get_required_external_files(df, path, features=None):
    """
    Checks if external files are required and if so, downloads them using the \
    Overpass API.

    Args:
        df (pandas.DataFrame): Contains points in order to define the area to \
            query with Overpass API
        path (str): Path to save the downloaded elements
        features (list, optional): Contains the names of the included features

    Returns:
        None
    """
    included_features = config.included_features if features is None else features
    required_args = set([
        arg for f in included_features
        for arg in features_getter_args_map[f]
    ])
    if 'street_gdf' in required_args:
        # osm_ut.extract_streets(get_centroids(df), path)
        osm_ut.extract_streets(get_points(df), path)
