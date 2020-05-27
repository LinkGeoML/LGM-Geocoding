import numpy as np
import pandas as pd
import json
import requests
from shapely.geometry import LineString
from sklearn.cluster import KMeans
import time
import os

from geocoding.config import Config


def query_api(query, fpath):
    """
    Queries Overpass API for *query*.

    Args:
        query (str): The query to be passed to API
        fpath (str): File path to write the API response

    Returns:
        None
    """
    status = 0
    overpass_url = 'http://overpass-api.de/api/interpreter'
    try:
        response = requests.get(overpass_url, params={'data': query}).json()
        with open(fpath, 'w') as f:
            json.dump(response, f)
    except ValueError:
        print('Overpass api error: Trying again with a greater timeout...')
        time.sleep(3)
        status = 1
    return status


def parse_streets(fpath):
    """
    Parses the API response from *fpath* and converts it to a dataframe.

    Args:
        fpath (str): File path to read

    Returns:
        pandas.DataFrame: Contains all streets as well as their geometries
    """
    # Helper function
    def convert_to_wkt_geometry(row):
        lons = [p['lon'] for p in row['geometry']]
        lats = [p['lat'] for p in row['geometry']]
        if len(lons) < 2 or len(lats) < 2:
            return None
        return LineString(list(zip(lons, lats)))

    with open(fpath, encoding='utf-8') as f:
        streets = json.load(f)['elements']
    if not streets:
        return None

    data = [(street['id'], street['geometry']) for street in streets]
    cols = ['id', 'geometry']
    street_df = pd.DataFrame(data=data, columns=cols)
    street_df['geometry'] = street_df.apply(convert_to_wkt_geometry, axis=1)
    street_df = street_df.dropna()
    return street_df


def extract_streets(points, path):
    """
    A wrapper function that administrates the streets download.

    Args:
        points (numpy.ndarray): Contains the data points that define the area \
            to extract from Overpass API
        path (str): Path to write

    Returns:
        None
    """
    labels = cluster_points(points)
    clusters_bboxes = get_clusters_bboxes(points, labels)
    street_dfs = []
    for cluster, bbox in clusters_bboxes.items():
        print('Getting bbox', cluster + 1, 'out of', len(clusters_bboxes))
        cell_street_df = download_cell(bbox, os.path.join(path, "osm_streets.json"))
        if cell_street_df is not None:
            print('Number of streets:', len(cell_street_df))
            street_dfs.append(cell_street_df)
        else:
            print('Number of streets:', 0)
        # if (cluster + 1) % 5 == 0:
        #     print(f'Suspended for {config.osm_timeout} secs...')
        #     time.sleep(config.osm_timeout)
    # delete file
    if os.path.exists(os.path.join(path, "osm_streets.json")):
        os.remove(os.path.join(path, "osm_streets.json"))

    street_df = pd.concat(street_dfs, ignore_index=True)
    street_df.drop_duplicates(subset='id', inplace=True)
    street_df.to_csv(f'{os.path.join(path, "osm_streets.csv")}', columns=['id', 'geometry'], index=False)
    print(f'Extracted {len(street_df.index)} unique streets')


def download_cell(cell, fpath):
    """
    Downloads *cell* from Overpass API, writes results in *fpath* and then \
    parses them into a pandas.DataFrame.

    Args:
        cell (list): Contains the bounding box coords
        fpath (str): Path to write results and then to read from in order to \
            parse them

    Returns:
        pandas.DataFrame: Contains all street elements included in *cell*
    """
    west, south, east, north = cell
    counter = 0
    status = 1
    while status and (counter < Config.max_overpass_tries):
        counter += 1
        query = (
            f'[out:json][timeout:{Config.osm_timeout * counter}];'        
            # f'way["highway"]["highway"!~"^(cycleway|footway)$"]'
            f'way["highway"]["highway"!~"^(cycleway)$"]'
            # 'way["highway"~"^(motorway|trunk|primary)$"];'
            # 'way["highway"]'
            f'({south},{west},{north},{east});'
            'out geom;')
        status = query_api(query, fpath)

    if status:
        print('Overpass api error: Exiting.')
        exit()
    return parse_streets(fpath)


def cluster_points(X):
    """
    Clusters points given in *X*.

    Args:
        X (numpy.ndarray): Contains the points to be clustered

    Returns:
        numpy.ndarray: The predicted clusters labels
    """
    n_clusters = int(Config.clusters_pct * X.shape[0])
    kmeans = KMeans(
        n_clusters=n_clusters, random_state=Config.seed_no, n_init=20, max_iter=500, n_jobs=Config.n_jobs
    ).fit(X)
    labels = kmeans.predict(X)
    return labels


def get_clusters_bboxes(X, labels):
    """
    Extracts a bounding box for each one of the clusters.

    Args:
        X (numpy.ndarray): Contains the clustered points
        labels (numpy.ndarray): Contains the cluster label for each point in \
            *X*
    Returns:
        dict: Contains the cluster labels as keys and the corresponding \
            bounding box as values
    """
    bboxes = {}
    for i in range(len(set(labels))):
        cluster_points = np.vstack([p for j, p in enumerate(X) if labels[j] == i])
        xmin, ymin = cluster_points.min(axis=0) - Config.osm_buffer
        xmax, ymax = cluster_points.max(axis=0) + Config.osm_buffer
        bboxes[i] = [xmin, ymin, xmax, ymax]
    # print({k: v for k, v in sorted(bboxes.items(), key=lambda item: item[1][0])})
    return bboxes
