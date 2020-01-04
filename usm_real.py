import os
import sys
import warnings
from functools import partial

from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, mutual_info_score

from mlrank.preprocessing.dichotomizer import dichotomize_vector
from mlrank.submodular.metrics import (
    log_likelihood_regularized_score_val,

)

from mlrank.submodular.optimization.ffs import ForwardFeatureSelectionExtended

import numpy as np
import pandas as pd

from sklearn import clone
from sklearn.externals import joblib
from sklearn.utils.multiclass import type_of_target

from itertools import product

from mlrank.submodular.optimization.multilinear import (
    MultilinearUSMExtended,
    MultilinearUSMClassic
)

from mlrank.benchmarks.holdout_bench import HoldoutBenchmark, DichtomizedHoldoutBenchmark

# models
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.neural_network import MLPRegressor, MLPClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

BREAST_CANCER_PATH = './datasets/breast_cancer.csv'

ADULT_TRAIN_PATH = './datasets/adult_train.csv'
ADULT_TEST_PATH = './datasets/adult_test.csv'

AMAZON_TRAIN_PATH = './datasets/amazon_train.csv'
AMAZON_TEST_PATH = './datasets/amazon_test.csv'

INTERNET_TRAIN_PATH = './datasets/internet_train.dat'
INTERNET_TEST_PATH = './datasets/internet_train.dat'

#ARRHYTHMIA_PATH = './datasets/arrhythmia.data'
#FOREST_FIRE_PATH = './datasets/forestfires.csv'
#HEART_DESEASE_PATH = './datasets/reprocessed.hungarian.data'
#SEIZURES_PATH = './datasets/seizures.csv'
#LUNG_CANCER_PATH = './datasets/lung-cancer.data'

# algorithm params
ALGO_PARAMS = {
    'decision_function': [
        {'regression': Lasso(),
         'classification': LogisticRegression(multi_class='auto', solver='liblinear', penalty='l2', C=1000)},
        #{'regression': MLPRegressor(hidden_layer_sizes=(5, 5), activation='relu'),
        # 'classification': MLPClassifier(hidden_layer_sizes=(5, 5), activation='relu')},
        #{'regression': LGBMRegressor(
        #        boosting_type='rf',
        #        learning_rate=1e-2,
        #        max_depth=5,
        #        subsample=0.7,
        #        n_estimators=200,
        #        verbose=-1,
        #        subsample_freq=5,
        #        num_leaves=2**5,
        #        silent=True
        #    ),
        #'classification': LGBMClassifier(
        #        boosting_type='rf',
        #        learning_rate=1e-2,
        #        max_depth=5,
        #        subsample=0.7,
        #        n_estimators=200,
        #        verbose=-1,
        #        subsample_freq=5,
        #        num_leaves=2 ** 5,
        #        silent=True
        #    )
        #}
    ]
}

# hyperparameters
HYPERPARAMS = {
    'bins': [2, 4, 8],
    'lambda': [.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
}


if __name__ == '__main__':
    np.random.seed(42)

    feature_selection_share=.5

    joblib.dump('test', "./data/testdoc.bin")

    results = {}

    for dataset, decision_function in product(ALGO_PARAMS['dataset'], ALGO_PARAMS['decision_function']):
        dfunc = decision_function[dataset['problem']]
        key = "{}, {}".format(dataset['name'], dfunc.__class__.__name__)
        results[key] = list()

        print('>>', key)
        
        X, y = dataset['data']

        prev_lambda_param = None

        for lambda_param, bins in product(HYPERPARAMS['lambda'], HYPERPARAMS['bins']):
            print(bins, lambda_param)
            if bins >= X.shape[0] * feature_selection_share:
                print(key, bins, 'very small dataset for such dichtomization.')
                continue

            score_function = partial(log_likelihood_regularized_score_val, _lambda=lambda_param)

            bench = DichtomizedHoldoutBenchmark(
                ForwardFeatureSelectionExtended(
                    decision_function=dfunc,
                    score_function=score_function,
                    n_bins=bins,
                    train_share=0.8,
                    n_cv_ffs=8,
                ),
                feature_selection_share=feature_selection_share,
                decision_function=dfunc,
                n_holdouts=70,
                n_bins=bins,
                n_jobs=8
            )

            predictions = bench.benchmark(X, y)

            results[key].append({
                'bins': bins,
                'lambda': lambda_param,
                'result': predictions
            })

            joblib.dump(results, "./data/mlrank_realdata_usm_lik_full_5.bin")
