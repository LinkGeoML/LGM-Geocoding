# LGM-Geocoding
A python library for accurate classification of best geocoding sources per coordinate pair.

## About LGM-Geocoding
LGM-Geocoding is a python library which implements a full Machine Learning workflow for training classification algorithms on annotated datasets that contain mappings between coordinate pairs and the ideal geocoding source for them and producing models for providing accurate predictions about ideal geocoding sources for previously unseen geocoding pairs. Geocoding implements a series of training features, regarding the different coordinate pairs that are available for each geocoder and the interaction between them and neighboring geospacial data (namely road data). Further, it encapsulates grid-search and cross-validation functionality, based on the [scikit](https://scikit-learn.org/) toolkit, assessing as series of classification models and parameterizations, in order to find the most fitting model for the data at hand.

## Dependencies
* python 3
* numpy
* pandas
* sklearn
* geopandas
* matplotlib
* psycopg2
* osmnx
* shapely
* argparse

## Instructions
In order for the library to function the user must provide it with a .csv file containing a collection of coordinate pairs. More specifically, in this iteration of the library the .csv file must at least contain exactly three coordinate pairs (i.e. six columns, each for each coordinate) that correspond to each point in the dataset and a column that refers to the label that corresponds to its annotated best geocoding source. The first six columns must have the names "X2", "Y2", "X3", "Y3" and "X4", "Y4". The label column must be named "dset".

**Algorithm evaluation/selection**: consists of an exhaustive comparison between several classification algorithms that are available in the scikit-learn library. Its purpose is to
compare the performance of every algorithm-hyperparameter configuration in a nested cross-validation scheme and produce the best candidate-algorithm for further usage. More specifically this step outputs three files:

* a file consisting of the algorithm and parameters space that was searched, 
* a file containing the results per cross-validation fold and their averages and
* a file containing the name of the best model.

You can execute this step as follows: ```python find_best_clf.py -geocoding_file_name <csv containing geocoding information> -results_file_name <desired name of the csv to contain the metric results per fold> -hyperparameter_file_name <desired name of the file to contain the hyperparameter space that was searched>```.

The last two arguments are optional and their values are defaulted to:
* classification_report_*timestamp*, and
* hyperparameters_per_fold_*timestamp*

correspondingly

**Algorithm tuning**: The purpose of this step is to further tune the specific algorithm that was chosen in step 1 by comparing its performance while altering the hyperparameters with which it is being configured. This step outputs the hyperparameter selection corresponding to the best model.

You can execute this step as follows: ```python finetune_best_clf.py -geocoding_file_name <csv containing geocoding information> -best_hyperparameter_file_name <desired name of the file to contain the best hyperparameters that were selected for the best algorithm of step 1> -best_clf_file_name <file containing the name of the best classifier>```.

All arguments except pois_csv_name are optional and their values are defaulted to:

* best_hyperparameters_*category level*_*timestamp*.csv
* the latest file with the *best_clf_* prefix

**Model training on a specific training set**: This step handles the training of the final model on an entire dataset, so that it can be used in future cases. It outputs a pickle file in which the model is stored.

You can execute this step as follows: ```python export_best_model.py -geocoding_file_name <csv containing geocoding information> -best_hyperparameter_file_name <csv containing best hyperparameter configuration for the classifier -best_clf_file_name <file containing the name of the best classifier> -trained_model_file_name <name of file where model must be exported>```.

All arguments except pois_csv_name are optional and their values are defaulted to:

* the latest file with the *best_hyperparameters_* prefix
* the latest file with the best_clf_* prefix
* trained_model_*level*_*timestamp*.pkl

correspondingly.

**Predictions on novel data**: This step can be executed as: ```python export_predictions.py -geocoding_file_name <csv containing geocoding information> -results_file_name <desired name of the output csv> -trained_model_file_name <pickle file containing an already trained model>```

The output .csv file will contain the k most probable predictions regarding the category of each POI. If no arguments for output_csv are given, their values are defaulted to:
* output_csv = predictions_*timestamp*.csv and 
* trained_model_file_name = *name of the latest produced pickle file in the working directory*.
