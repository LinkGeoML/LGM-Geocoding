#!/usr/bin/python

# import the necessary packages
import geopandas as gpd
import psycopg2
import argparse
import numpy as np
import nltk
import itertools
import random

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split

import csv
import datetime
import config
import glob
import os

from geocoding_feature_extraction import *

np.random.seed(1234)

def get_scores(X_test, y_test, clf):
	
	y_pred = clf.predict(X_test)	
	
	#print(X_test.shape)
	probs = clf.predict_proba(X_test)
	
	return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted'), f1_score(y_test, y_pred, average='macro')

def fine_tune_parameters_given_clf(clf_name, X_train, y_train, X_test, y_test):
	
	scores = ['accuracy']
	
	if clf_name == "SVM":
		tuned_parameters = config.initialConfig.SVM_hyperparameters
		clf = SVC(probability=True)
		
	elif clf_name == "Nearest Neighbors":
		tuned_parameters = config.initialConfig.kNN_hyperparameters
		clf = KNeighborsClassifier()
		
	elif clf_name == "Decision Tree":

		tuned_parameters = config.initialConfig.DecisionTree_hyperparameters
		clf = DecisionTreeClassifier()
		
	elif clf_name == "Random Forest":

		tuned_parameters = config.initialConfig.RandomForest_hyperparameters
		clf = RandomForestClassifier()
		
	elif clf_name == "Extra Trees":

		tuned_parameters = config.initialConfig.RandomForest_hyperparameters
		clf = ExtraTreesClassifier()
	
	elif clf_name == "MLP":
		
		tuned_parameters = config.initialConfig.MLP_hyperparameters
		clf = MLPClassifier()
	
	"""
	elif clf_name == "AdaBoost":
		tuned_parameters = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "n_estimators": [1, 2]
             }
		clf = AdaBoostClassifier()
	
	elif clf_name == "MLP":
		tuned_parameters = {'hidden_layer_sizes': [(256,), (512,), (128, 256, 128,)]}
		clf = MLPClassifier()
		
	elif clf_name == "Gaussian Process":
		
		clf = GaussianProcessClassifier()
	
	elif clf_name == "QDA":
		tuned_parameters = 
		clf = QuadraticDiscriminantAnalysis()
	"""
	
	print(clf_name)
	
	print(X_train.shape, y_train.shape)
		
	for score in scores:

		clf = GridSearchCV(clf, tuned_parameters, cv=4,
						   scoring='%s' % score, verbose=0)
		clf.fit(X_train, y_train)
		
	return clf

def tuned_parameters_5_fold(args):
	
	# Shuffle ids
	
	#X, y = get_X_Y_data()
	X = np.loadtxt('X.csv', delimiter=",")
	y = np.loadtxt('y.csv', delimiter=",")
			
	clf_names_not_tuned = ["Naive Bayes", "MLP", "Gaussian Process", "QDA", "AdaBoost"]
	clf_names = config.initialConfig.classifiers
	clf_scores_dict = dict.fromkeys(clf_names)
	for item in clf_scores_dict:
		clf_scores_dict[item] = [[], [], []]
	report_data = []
	hyperparams_data = []
			
	# get train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
	print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
	
	X_train, scaler = standardize_data_train(X)
	X_test = standardize_data_test(X_test, scaler)
			
	# read clf name from csv
	row = {}
	
	if args['best_clf'] in clf_names_not_tuned:
		if clf_name == "Naive Bayes":
			clf = GaussianNB()
			clf.fit(X_train, y_train)
		#elif clf_name == "MLP":
		#	clf = MLPClassifier()
		#	clf.fit(X_train, y_train)
		elif clf_name == "Gaussian Process":
			clf = GaussianProcessClassifier()
			clf.fit(X_train, y_train)
		#elif clf_name == "QDA":
		#	clf = QuadraticDiscriminantAnalysis()
		#	clf.fit(X_train, y_train)
		else:
			clf = AdaBoostClassifier()
			clf.fit(X_train, y_train)
	else:
		clf = fine_tune_parameters_given_clf(args['best_clf'], X_train, y_train, X_test, y_test)
	
	hyperparams_data = clf.best_params_
	#print(hyperparams_data)
	df2 = pd.DataFrame.from_dict([hyperparams_data])
	
	#print(df2)
	"""
	if args['best_hyperparameter_file_name'] is not None:
		filename = args['best_hyperparameter_file_name'] + '_' + str(datetime.datetime.now()) + '.csv'
	else:
		filename = 'best_hyperparameters_' + str(datetime.datetime.now()) + '.csv'
	df2.to_csv(filename, index = False)
	"""
	
	if config.initialConfig.experiment_folder == None:
		experiment_folder_path = config.initialConfig.root_path + 'experiment_folder_*'
		list_of_folders = glob.glob(experiment_folder_path)
		if list_of_folders == []:
			print("ERROR! No experiment folder found inside the root folder")
			return
		else:
			latest_experiment_folder = max(list_of_folders, key=os.path.getctime)
			filepath = latest_experiment_folder + '/' + 'best_hyperparameters.csv'
			df2.to_csv(filepath, index = False)
	else:
		experiment_folder_path = config.initialConfig.root_path + config.initialConfig.experiment_folder
		filepath = experiment_folder_path + '/' + 'best_hyperparameters.csv'
		df2.to_csv(filepath, index = False)
	
def main():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-geocoding_file_name", "--geocoding_file_name", required=False,
		help="name of table containing pois information")
	#ap.add_argument("-results_file_name", "--results_file_name", required=False,
	#	help="desired name of best hyperparameter file")
	ap.add_argument("-best_hyperparameter_file_name", "--best_hyperparameter_file_name", required=False,
		help="desired name of best hyperparameter file")
	ap.add_argument("-best_clf_file_name", "--best_clf_file_name", required=False,
		help="name of file containing the best classifier found in step 1")

	args = vars(ap.parse_args())

	"""
	if args['best_clf_file_name'] is not None:
		#with open(args['best_clf_file_name']) as f:
		#	args['best_clf'] = f.readline()
		input_file = csv.DictReader(open(args['best_clf_file_name']))
		with open(input_file, 'r') as csv_file:
			reader = csv.reader(csv_file)
			count = 0
			for row in reader:
				if count == 1:
					args['best_clf'] = row[0]
				count += 1
	else:
		list_of_files = glob.glob('best_clf_*')
		input_file = max(list_of_files, key=os.path.getctime)
		with open(input_file, 'r') as csv_file:
			reader = csv.reader(csv_file)
			count = 0
			for row in reader:
				if count == 1:
					args['best_clf'] = row[0]
				count += 1
	"""
	
	if config.initialConfig.experiment_folder is not None:
		experiment_folder_path = config.initialConfig.root_path + config.initialConfig.experiment_folder
		exists = os.path.isdir(experiment_folder_path)
		if exists:
			filepath = experiment_folder_path + '/' + 'best_clf.csv'
			exists2 = os.path.isfile(filepath)
			if exists2:
				with open(filepath, 'r') as csv_file:
					reader = csv.reader(csv_file)
					count = 0
					for row in reader:
						if count == 1:
							args['best_clf'] = row[0]
						count += 1
			else:
				print("ERROR! No best_clf file found inside the folder")
				return
		else:
			print("ERROR! No experiment folder with the given name found")
			return
	else:
		experiment_folder_path = config.initialConfig.root_path + 'experiment_folder_*'
		list_of_folders = glob.glob(experiment_folder_path)
		if list_of_folders == []:
			print("ERROR! No experiment folder found inside the root folder")
			return
		else:
			latest_experiment_folder = max(list_of_folders, key=os.path.getctime)
			filepath = latest_experiment_folder + '/' + 'best_clf.csv'
			exists = os.path.isfile(filepath)
			if exists:
				with open(filepath, 'r') as csv_file:
					reader = csv.reader(csv_file)
					count = 0
					for row in reader:
						if count == 1:
							args['best_clf'] = row[0]
						count += 1
			else:
				print("ERROR! No best_clf file found inside the folder!")	
				return
			
	tuned_parameters_5_fold(args)
	
if __name__ == "__main__":
   main()
