import pandas as pd
import numpy as np
import osmnx as ox
import shapely
import geopandas as gpd
from shapely.geometry import Point
import argparse
import config
import os


def get_X_Y_data():

	#in_df = pd.read_excel("geocoding_test.xlsx")
	in_df = pd.read_excel("Geocoding.xlsx")
	
	labels = in_df['dset']
	#print(in_df)
	
	feature_dict = dict((el, None) for el in config.initialConfig.feature_list)
	for key in config.initialConfig.included_features:
		feature_dict[key] = []
	
	sentinel = 0
	for key in config.initialConfig.included_features:

		filepath = config.initialConfig.root_path + key + '.csv'
		exists = os.path.isfile(filepath)
		if exists:
			if sentinel == 0:
				total_features = np.genfromtxt(filepath, delimiter=',')
				sentinel = 1
			else:
				temp_array = np.genfromtxt(filepath, delimiter=',')
				total_features = np.concatenate((total_features, temp_array), axis = 1)
				#sentinel = 1
				
	sentinel = 0
	for key in config.initialConfig.included_features:
		filepath = config.initialConfig.root_path + key + '.csv'
		exists = os.path.isfile(filepath)
		if exists:
			if sentinel == 0:
				total_features = np.genfromtxt(filepath, delimiter=',')
				sentinel = 1
			else:
				temp_array = np.genfromtxt(filepath, delimiter=',')
				total_features = np.concatenate((total_features, temp_array), axis = 1)
				#sentinel = 1
	#print(total_features.shape)
	
	if not exists:
		ids = in_df['ID']

		X2 = in_df['X2']
		Y2 = in_df['Y2']
		X3 = in_df['X3']
		Y3 = in_df['Y3']
		X4 = in_df['X4']
		Y4 = in_df['Y4']

		offset1 = 0
		offset2 = Y4.shape[0]
		pairwise_distances = [[] for _ in range(offset1, offset2)]
		pairwise_distances_single_coordinate = [[] for _ in range(offset1, offset2)]
		average_distances_from_centroid = [[] for _ in range(offset1, offset2)]
		distances_from_centroid = [[] for _ in range(offset1, offset2)]
		distances_from_closest_street = [[] for _ in range(offset1, offset2)]
		normalized_features = [[] for _ in range(offset1, offset2)]

		total_features_u = [[] for _ in range(offset1, offset2)]
		total_features = [[] for _ in range(offset1, offset2)]

		X2_array = np.asarray(X2)
		Y2_array = np.asarray(Y2)
		X3_array = np.asarray(X3)
		Y3_array = np.asarray(Y3)
		X4_array = np.asarray(X4)
		Y4_array = np.asarray(Y4)

		#print(X2_array)
		X2_mean = np.mean(X2_array)
		X2_var = np.var(X2_array)
		Y2_mean = np.mean(Y2_array)
		Y2_var = np.var(Y2_array)
		X3_mean = np.mean(X3_array)
		X3_var = np.var(X3_array)
		Y3_mean = np.mean(Y3_array)
		Y3_var = np.var(Y3_array)
		X4_mean = np.mean(X4_array)
		X4_var = np.var(X4_array)
		Y4_mean = np.mean(Y4_array)
		Y4_var = np.var(Y4_array)

		#print(X2_mean, X2_var)

		for i in range(offset1, offset2):
			#print(ids[i])
			point1 = np.asarray([X2[i], Y2[i]])
			point2 = np.asarray([X3[i], Y3[i]])
			point3 = np.asarray([X4[i], Y4[i]])
			
			if feature_dict['normalized_features'] is not None:
				normalized_features[i-offset1].append(np.abs(X2[i] - X2_mean) / X2_var)
				normalized_features[i-offset1].append(np.abs(Y2[i] - Y2_mean) / Y2_var)
				normalized_features[i-offset1].append(np.abs(X3[i] - X3_mean) / X3_var)
				normalized_features[i-offset1].append(np.abs(Y3[i] - Y3_mean) / Y3_var)
				normalized_features[i-offset1].append(np.abs(X4[i] - X4_mean) / X4_var)
				normalized_features[i-offset1].append(np.abs(Y4[i] - Y4_mean) / Y4_var)
			
			if feature_dict['pairwise_distances'] is not None:
				pairwise_distances[i-offset1].append(np.linalg.norm(point1 - point2))
				pairwise_distances[i-offset1].append(np.linalg.norm(point1 - point3))
				pairwise_distances[i-offset1].append(np.linalg.norm(point2 - point3))
			
			if feature_dict['pairwise_distances_single_coordinate'] is not None:
				pairwise_distances_single_coordinate[i-offset1].append(np.abs(X2[i] - X3[i]))
				pairwise_distances_single_coordinate[i-offset1].append(np.abs(Y2[i] - Y3[i]))
				pairwise_distances_single_coordinate[i-offset1].append(np.abs(X2[i] - X4[i]))
				pairwise_distances_single_coordinate[i-offset1].append(np.abs(Y2[i] - Y4[i]))
				pairwise_distances_single_coordinate[i-offset1].append(np.abs(X3[i] - X4[i]))
				pairwise_distances_single_coordinate[i-offset1].append(np.abs(Y3[i] - Y4[i]))
			
			centroid_X = (X2[i] + X3[i] + X4[i]) / 3.0
			centroid_Y = (Y2[i] + Y3[i] + Y4[i]) / 3.0
			
			distance_X2_centroid = np.linalg.norm(np.abs(X2[i] - centroid_X))
			distance_Y2_centroid = np.linalg.norm(np.abs(Y2[i] - centroid_Y))
			distance_X3_centroid = np.linalg.norm(np.abs(X3[i] - centroid_X))
			distance_Y3_centroid = np.linalg.norm(np.abs(Y3[i] - centroid_Y))
			distance_X4_centroid = np.linalg.norm(np.abs(X4[i] - centroid_X))
			distance_Y4_centroid = np.linalg.norm(np.abs(Y4[i] - centroid_Y))
			
			if feature_dict['distances_from_centroid'] is not None:
				distances_from_centroid[i-offset1].append(distance_X2_centroid)
				distances_from_centroid[i-offset1].append(distance_Y2_centroid)
				distances_from_centroid[i-offset1].append(distance_X3_centroid)
				distances_from_centroid[i-offset1].append(distance_Y3_centroid)
				distances_from_centroid[i-offset1].append(distance_X4_centroid)
				distances_from_centroid[i-offset1].append(distance_Y4_centroid)
			
			if feature_dict['average_distances_from_centroid'] is not None:
				average_X_distance = (distance_X2_centroid + distance_X3_centroid + distance_X4_centroid) / 3.0
				average_Y_distance = (distance_Y2_centroid + distance_Y3_centroid + distance_Y4_centroid) / 3.0
				average_distances_from_centroid[i-offset1].append(average_X_distance)
				average_distances_from_centroid[i-offset1].append(average_Y_distance)
			
			if feature_dict['distances_from_closest_street'] is not None:
				north, south, east, west = centroid_Y + 0.01, centroid_Y - 0.01, centroid_X + 0.01, centroid_X - 0.01
				g = ox.graph_from_bbox(north, south, east, west, timeout = 360)
				gdf_nodes, street_df = ox.graph_to_gdfs(g)
				while(g is None):
					north += 0.01
					south += 0.01
					east += 0.01
					west += 0.01
					g = ox.graph_from_bbox(north, south, east, west) 
				gdf_nodes, street_df = ox.graph_to_gdfs(g)

				min_distance = 1000000000.0
				point1 = (X2[i], Y2[i])
				poi_geom1 = Point(point1)
				
				for index, row in street_df.iterrows():
					street_geom = row['geometry']
					if street_geom.distance(poi_geom1) < min_distance:
						min_distance = street_geom.distance(poi_geom1)
				distances_from_closest_street[i-offset1].append(min_distance)
				
				min_distance = 1000000000.0
				point2 = (X3[i], Y3[i])
				poi_geom2 = Point(point2)
				
				for index, row in street_df.iterrows():
					street_geom = row['geometry']
					if street_geom.distance(poi_geom2) < min_distance:
						min_distance = street_geom.distance(poi_geom2)
				distances_from_closest_street[i-offset1].append(min_distance)
				
				min_distance = 1000000000.0
				point3 = (X4[i], Y4[i])
				poi_geom3 = Point(point3)
				
				for index, row in street_df.iterrows():
					street_geom = row['geometry']
					if street_geom.distance(poi_geom3) < min_distance:
						min_distance = street_geom.distance(poi_geom3)
				distances_from_closest_street[i-offset1].append(min_distance)
				
			if feature_dict['normalized_features'] is not None:
				feature_dict['normalized_features'].append(normalized_features[i-offset1])
			if feature_dict['pairwise_distances'] is not None:
				feature_dict['pairwise_distances'].append(pairwise_distances[i-offset1])
			if feature_dict['pairwise_distances_single_coordinate'] is not None:
				feature_dict['pairwise_distances_single_coordinate'].append(pairwise_distances_single_coordinate[i-offset1])
			if feature_dict['distances_from_centroid'] is not None:
				feature_dict['distances_from_centroid'].append(distances_from_centroid[i-offset1])
			if feature_dict['average_distances_from_centroid'] is not None:
				feature_dict['average_distances_from_centroid'].append(average_distances_from_centroid[i-offset1])
			if feature_dict['distances_from_closest_street'] is not None: 
				feature_dict['distances_from_closest_street'].append(distances_from_closest_street[i-offset1])
			
			#print(ids[i])
			#total_features_u[i-offset1] = pairwise_distances[i-offset1] + pairwise_distances_single_coordinate[i-offset1] + average_distances_from_centroid[i-offset1] + distances_from_centroid[i-offset1] + distances_from_closest_street[i-offset1] + normalized_features[i-offset1]
			#total_features_u[i-offset1] = np.asarray(total_features_u[i-offset1])
		sentinel = 0
		for key in feature_dict:
			if feature_dict[key] is not None:
				if sentinel == 0:
					total_features = np.asarray(feature_dict[key])
					total_features, _ = standardize_data_train(total_features)
					filepath = config.initialConfig.root_path + key + '.csv'
					np.savetxt(filepath, total_features, delimiter=",")
					print("Feature Name: {0}, Mean Value: {1}, Std Value: {2}, Max Value: {3}, Min Value: {4}, Shape: {5}".format(key, np.mean(total_features), np.std(total_features), np.amax(total_features), np.amin(total_features), total_features.shape))
					sentinel = 1
				else:
					temp_array = np.asarray(feature_dict[key])
					temp_array, _ = standardize_data_train(temp_array)
					filepath = config.initialConfig.root_path + key + '.csv'
					np.savetxt(filepath, temp_array, delimiter=",")
					total_features = np.concatenate((total_features, temp_array), axis = 1)
					print("Feature Name: {0}, Mean Value: {1}, Std Value: {2}, Max Value: {3}, Min Value: {4}, Shape: {5}".format(key, np.mean(temp_array), np.std(temp_array), np.amax(temp_array), np.amin(temp_array), temp_array.shape))

		#total_features = np.vstack(total_features_u)
		print(total_features.shape)
	
	return total_features, labels
	
def standardize_data_train(X):
	from sklearn.preprocessing import MinMaxScaler
	
	standard_scaler = MinMaxScaler()
	X = standard_scaler.fit_transform(X)
	
	return X, standard_scaler

def standardize_data_test(X, scaler):
	from sklearn.preprocessing import MinMaxScaler
	
	X = standard_scaler.transform(X)
	
	return X
	
def main():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-geocoding_file_name", "--geocoding_file_name", required=False,
	help="name of table containing pois information")

	args = vars(ap.parse_args())
	X, y = get_X_Y_data()
	
	X, _ = standardize_data_train(X)
	

if __name__ == "__main__":
   main()
		
		
