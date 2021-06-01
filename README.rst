|MIT|

==================
LGM-Geocoding
==================

A python library for accurate classification of best geocoding sources per coordinate pair.

LGM-Geocoding is a python library which implements a full Machine Learning workflow for training classification algorithms
on annotated datasets that contain mappings between coordinate pairs and the ideal geocoding source for them and producing
models that provide accurate predictions about ideal geocoding sources for previously unseen geocoding pairs. Geocoding
implements a series of training features, regarding the different coordinate pairs that are available for each geocoder,
the interaction between them and neighboring geospacial data (namely road data). Further, it encapsulates grid-search
and cross-validation functionality, based on the `scikit-learn <https://scikit-learn.org/>`_ toolkit, assessing as series
of classification models and parameterizations, in order to find the most fitting model for the data at hand. Indicatively,
we succeed a 66.04% accuracy with the Random Forest classifier compared to the baseline approach that achieves
accuracy of 55.05% (see `References`_).

The source code was tested using Python 3.7 and Scikit-Learn 0.23.1 on a Linux server.

Description
-----------

The module consists of the following steps:

1. **Features extraction**
    This step takes into account the features configuration given in `config.py <https://github.com/LinkGeoML/LGM-Geocoding/
    geocoding/config.py>`_ and creates X_train and X_test feature pairs grouped by folds and ready to be utilized
    by machine learning algorithms in the next steps.

2. **Algorithm selection**
    A list of given classifiers given in `config.py <https://github.com/LinkGeoML/LGM-Geocoding/geocoding/config.py>`_
    are evaluated in a nested cross-validation scheme in order to find which performs the best on the features sets created
    in the previous step.

3. **Model selection**
    Given a selected classifier, this step tries to find its best configuration.

4. **Model training**
    Utilizing the knowledge from the previous step, a model is trained on the whole available data using the optimal
    configuration. This model is then saved to disk for later usage.

5. **Model deployment**
    This step loads the optimal model from disk and uses it in order to classify a set of unseen, unlabeled, test data.
    Classification results come in a form of (predicted-label, score) pairs, suggesting the model's confidence about each
    prediction.

Setup procedure
---------------

Download the latest version from the `GitHub repository <https://github.com/LinkGeoML/LGM-Geocoding.git>`_, change to
the main directory and run:

.. code:: bash

   pip install -r pip_requirements.txt

It should install all the required libraries automatically (*scikit-learn, numpy, pandas etc.*).

Usage
-----
The execution of the project starts with the **Features extraction** step initializing the pipeline's root folder which
the following steps will refer to in order to output their results. Each step can be executed as follows:

1. **Features extraction**
    .. code:: bash

        python -m geocoding.features_extraction -fpath <fpath>

    where *<fpath>* is the path to the csv file containing train data.

2. **Algorithm selection**
    .. code:: bash

        python -m geocoding.algorithm_selection -experiment_path <exp_path>

    where *<exp_path>* is the path to the folder created from the first step.

3. **Model selection**
    .. code:: bash

        python -m geocoding.model_selection -classifier <clf_name> -experiment_path <exp_path>

    where *<clf_name>* is the classifier's name to be optimized in order to build the model and *<exp_path>* same as before.

4. **Model training**
    .. code:: bash

        python -m geocoding.model_training -experiment_path <exp_path>

    where *<exp_path>* same as before.

5. **Model deployment**
    .. code:: bash

        python -m geocoding.model_deployment -experiment_path <exp_path> -fpath <fpath>

    where *<exp_path>* same as before and *<fpath>* is the path to the csv file containing the test data.

Documentation
-------------
Source code documentation is available from `linkgeoml.github.io`__.

__ https://linkgeoml.github.io/LGM-Geocoding/

References
----------
* K. Alexis et al. Improving geocoding quality via learning to integrate multiple geocoders. SSDBM ’20.

License
-------
LGM-Geocoding is available under the `MIT <https://opensource.org/licenses/MIT>`_ License.

..
    .. |Documentation Status| image:: https://readthedocs.org/projects/coala/badge/?version=latest
       :target: https://linkgeoml.github.io/LGM-Interlinking/

.. |MIT| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT

