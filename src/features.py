import numpy as np
from itertools import combinations
from shapely.geometry import Point, asLineString
import re

from src import features_utilities as feat_ut
from src.config import config


def get_normalized_coords(df):
    """
    Creates a features array. Normalizes each longitude or latitude column, \
    by subtracting the corresponding column's mean value from it.

    Args:
        df (pandas.DataFrame): Contains data points for which the features \
            will be created

    Returns:
        numpy.ndarray: The features array of shape (n_samples, n_features), \
            here (len(df), number_of_services * 2)
    """
    cols = [
        col for service in config.services
        for col in (f'x_{service}', f'y_{service}')
    ]
    X = np.zeros((len(df), len(cols)))
    for i, col in enumerate(cols):
        X[:, i] = df[col] - df[col].mean()
    return X


def get_pairwise_coords_distances(df):
    """
    Creates a features array. For each address (each row), calculates the \
    pairwise distances among coordinates suggested from different services.

    Args:
        df (pandas.DataFrame): Contains data points for which the features \
            will be created

    Returns:
        numpy.ndarray: The features array of shape (n_samples, n_features), \
            here (len(df), number_of_services * (number_of_services-1))
    """
    n_services = len(config.services)
    X = np.zeros((len(df), n_services*(n_services-1)))
    for i in df.itertuples():
        distances = []
        for coord in ['lon', 'lat']:
            coords = [df.loc[i.Index, f'{coord}_{service}']
                      for service in config.services]
            pairs = combinations(coords, 2)
            coords_distances = [np.abs(pair[0]-pair[1]) for pair in pairs]
            distances.extend(coords_distances)
        X[i.Index] = feat_ut.filter(distances)
    return X


def get_pairwise_points_distances(df):
    """
    Creates a features array. For each address (each row), calculates the \
    pairwise distances among points suggested from different services.

    Args:
        df (pandas.DataFrame): Contains data points for which the features \
            will be created

    Returns:
        numpy.ndarray: The features array of shape (n_samples, n_features), \
            here (len(df), (number_of_services * (number_of_services-1)) / 2)
    """
    n_services = len(config.services)
    X = np.zeros((len(df), int((n_services*(n_services-1))/2)))
    for i in df.itertuples():
        points = [(df.loc[i.Index, f'lon_{service}'],
                   df.loc[i.Index, f'lat_{service}'])
                  for service in config.services]
        pairs = combinations(points, 2)
        distances = [
            Point(pair[0]).distance(Point(pair[1]))
            for pair in pairs
        ]
        X[i.Index] = feat_ut.filter(distances)
    return X


def get_centroid_coords_distances(df):
    """
    Creates a features array. For each address (each row), calculates the \
    distances between the corresponding centroid coords and the coords \
    suggested from different services.

    Args:
        df (pandas.DataFrame): Contains data points for which the features \
            will be created

    Returns:
        numpy.ndarray: The features array of shape (n_samples, n_features), \
            here (len(df), number_of_services * 2)
    """
    n_services = len(config.services)
    X = np.zeros((len(df), 2*n_services))
    for i in df.itertuples():
        distances = []
        for coord in ['lon', 'lat']:
            coords = [df.loc[i.Index, f'{coord}_{service}']
                      for service in config.services]
            coords_mean = np.mean(coords)
            coords_distances = [np.abs(c-coords_mean) for c in coords]
            distances.extend(coords_distances)
        X[i.Index] = feat_ut.filter(distances)
    return X


def get_centroid_points_distances(df):
    """
    Creates a features array. For each address (each row), calculates the \
    distances between the corresponding centroid and the points suggested \
    from different services.

    Args:
        df (pandas.DataFrame): Contains data points for which the features \
            will be created

    Returns:
        numpy.ndarray: The features array of shape (n_samples, n_features), \
            here (len(df), number_of_services)
    """
    n_services = len(config.services)
    X = np.zeros((len(df), n_services))
    for i in df.itertuples():
        points = [(df.loc[i.Index, f'lon_{service}'],
                   df.loc[i.Index, f'lat_{service}'])
                  for service in config.services]
        centroid = [sum(x)/len(x) for x in zip(*points)]
        distances = [
            Point(point).distance(Point(centroid))
            for point in points
        ]
        X[i.Index] = feat_ut.filter(distances)
    return X


def get_mean_centroids_coords_distances(df):
    """
    Creates a features array. For each address (each row), calculates the \
    mean distances between the corresponding centroid coords and the coords \
    suggested from different services.

    Args:
        df (pandas.DataFrame): Contains data points for which the features \
            will be created

    Returns:
        numpy.ndarray: The features array of shape (n_samples, n_features), \
            here (len(df), 2)
    """
    X = np.zeros((len(df), 2))
    for i in df.itertuples():
        distances = []
        for coord in ['lon', 'lat']:
            coords = [df.loc[i.Index, f'{coord}_{service}']
                      for service in config.services]
            coords_mean = np.mean(coords)
            coords_distances = [np.abs(c-coords_mean) for c in coords]
            distances.append(np.mean(coords_distances))
        X[i.Index] = feat_ut.filter(distances)
    return X


def get_mean_centroids_points_distances(df):
    """
    Creates a features array. For each address (each row), calculates the \
    mean distance between the corresponding centroid and the points \
    suggested from different services.

    Args:
        df (pandas.DataFrame): Contains data points for which the features \
            will be created

    Returns:
        numpy.ndarray: The features array of shape (n_samples, n_features), \
            here (len(df), 1)
    """
    X = np.zeros((len(df), 1))
    for i in df.itertuples():
        points = [(df.loc[i.Index, f'lon_{service}'],
                   df.loc[i.Index, f'lat_{service}'])
                  for service in config.services]
        centroid = [sum(x)/len(x) for x in zip(*points)]
        distances = [
            Point(point).distance(Point(centroid))
            for point in points
        ]
        X[i.Index] = np.mean(feat_ut.filter(distances))
    return X


def get_nearest_street_distance_per_service(df, street_gdf):
    """
    Creates a features array. For each address (each row) and for each \
    service, calculates the distance to the nearest street.

    Args:
        df (pandas.DataFrame): Contains data points for which the features \
            will be created
        street_gdf (geopandas.GeoDataFrame): Contains all streets extracted \
            from OSM, along with their geometries

    Returns:
        numpy.ndarray: The features array of shape (n_samples, n_features), \
            here (len(df), number_of_services)
    """
    street_index = street_gdf.sindex
    n_services = len(config.services)
    X = np.zeros((len(df), n_services))
    for i in df.itertuples():
        points = [(df.loc[i.Index, f'lon_{service}'],
                   df.loc[i.Index, f'lat_{service}'])
                  for service in config.services]
        distances = [
            min([
                Point(p).distance(street_gdf.iloc[c]['geometry'])
                for c in list(street_index.nearest(p))
            ]) for p in points
        ]
        # print(distances, points)
        # print([
        #     Point(p).distance(street_gdf.iloc[c]['geometry'])
        #     for p in points
        #     for c in list(street_index.nearest(p))
        # ])
        # print([
        #     street_gdf.iloc[c]['geometry'].wkt for p in points for c in list(street_index.nearest(p))
        # ])
        X[i.Index] = feat_ut.filter(distances)
    return X


def get_common_nearest_street_distance(df, street_gdf, k=3):
    """
    Creates a features array. For each address (each row) and for each \
    service, calculates the distance to the nearest street.

    Args:
        df (pandas.DataFrame): Contains data points for which the features \
            will be created
        street_gdf (geopandas.GeoDataFrame): Contains all streets extracted \
            from OSM, along with their geometries

    Returns:
        numpy.ndarray: The features array of shape (n_samples, n_features), \
            here (len(df), number_of_services)
    """
    street_index = street_gdf.sindex
    n_services = len(config.services)
    X = np.zeros((len(df), n_services))
    for i in df.itertuples():
        points = [
            (df.loc[i.Index, f'lon_{service}'], df.loc[i.Index, f'lat_{service}']) for service in config.services
        ]
        lines = [
            [street_gdf.iloc[c]['geometry'] for c in list(street_index.nearest(p, k))] for p in points
        ]

        common_lines = []
        for li in range(1, len(lines)):
            for l1, l2 in ((x, y) for x in lines[0] for y in lines[li]):
                if l1.intersects(l2):
                    common_lines.append(list(l1.coords))

        if len(common_lines):
            mask = np.ones(len(common_lines))
            combs = combinations(enumerate(common_lines), 2)
            for c in combs:
                if asLineString(c[0][1]).intersects(asLineString(c[1][1])): mask[c[0][0]] = 0

            masked_clines = np.ma.masked_array(np.arange(len(common_lines)), mask=mask)
            if len(masked_clines.compressed()):
                distances = [
                    np.mean([Point(p).distance(asLineString(common_lines[c])) for c in masked_clines.compressed()]) for p in points
                ]
            else:
                closer_geom = min(
                    [(idx, Point(points[0]).distance(asLineString(c))) for idx, c in enumerate(common_lines)],
                    key=lambda x: x[1]
                )
                distances = [Point(p).distance(asLineString(common_lines[closer_geom[0]])) for p in points]
        else:
            closer_geom = min(
                [(idx, Point(points[0]).distance(c)) for idx, c in enumerate(lines[0])],
                key=lambda x: x[1]
            )
            distances = [Point(p).distance(lines[0][closer_geom[0]]) for p in points]

        X[i.Index] = feat_ut.filter(distances)
    return X


def get_nearest_street_distance_by_centroid(df, street_gdf):
    """
    Creates a features array. For each address (each row), the nearest street \
    to the corresponding centroid is identified at first. Then, distances \
    between this street and points suggested from different services are \
    calculated.

    Args:
        df (pandas.DataFrame): Contains data points for which the features \
            will be created
        street_gdf (geopandas.GeoDataFrame): Contains all streets extracted \
            from OSM, along with their geometries

    Returns:
        numpy.ndarray: The features array of shape (n_samples, n_features), \
            here (len(df), number_of_services)
    """
    street_index = street_gdf.sindex
    n_services = len(config.services)
    X = np.zeros((len(df), n_services))
    for i in df.itertuples():
        points = [(df.loc[i.Index, f'lon_{service}'],
                   df.loc[i.Index, f'lat_{service}'])
                  for service in config.services]
        centroid = [sum(x)/len(x) for x in zip(*points)]
        candidates = list(street_index.nearest(centroid))
        nearest = candidates[np.argmin([
            Point(centroid).distance(street_gdf.iloc[c]['geometry'])
            for c in candidates
        ])]
        distances = [
            Point(p).distance(street_gdf.iloc[nearest]['geometry'])
            for p in points
        ]
        X[i.Index] = feat_ut.filter(distances)
    return X


def get_zip_codes(df):
    """
    Creates a features array. For each address (each row), the first 2 digits \
    of its zip code are extracted. Then for each row *r*, the array will \
    contain 1 (True) in column *c*, if *c* represents the 2 digits that *r*'s \
    zip code starts with.

    Args:
        df (pandas.DataFrame): Contains data points for which the features \
            will be created

    Returns:
        numpy.ndarray: The features array of shape (n_samples, n_features), \
            here (len(df), 76). This is due to the fact that there are 76 \
            such valid combinations in Greece
    """
    pattern = r'\d{5}(?=,)'
    X = np.zeros((len(df), 85-10+1))
    for i in df.itertuples():
        zip_code = re.search(pattern, i.address).group(0)
        region_code = zip_code[:2]
        X[i.Index, int(region_code)-10] = 1
    return X
