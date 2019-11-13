import os
import sys
import warnings
from functools import partial

from mlrank.preprocessing.dichtomizer import DichtomizationImpossible
from mlrank.submodular.metrics import log_likelihood_regularized_score_val
from mlrank.submodular.optimization.ffs import ForwardFeatureSelectionClassic, ForwardFeatureSelectionExtended

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

import numpy as np
from sklearn.externals import joblib
from itertools import product
from mlrank.benchmarks.holdout import DichtomizedHoldoutBenchmark
from mlrank.benchmarks.traintest import TrainTestBenchmark, DichtomizedTrainTestBenchmark
from mlrank.datasets import (AdultDataSet, AmazonDataSet, BreastDataSet)

# models
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mutual_info_score


BREAST_CANCER_PATH = './datasets/breast_cancer.csv'

ADULT_TRAIN_PATH = './datasets/adult_train.csv'
ADULT_TEST_PATH = './datasets/adult_test.csv'

AMAZON_TRAIN_PATH = './datasets/amazon_train.csv'
AMAZON_TEST_PATH = './datasets/amazon_test.csv'

#ARRHYTHMIA_PATH = './datasets/arrhythmia.data'
#FOREST_FIRE_PATH = './datasets/forestfires.csv'
#HEART_DESEASE_PATH = './datasets/reprocessed.hungarian.data'
#SEIZURES_PATH = './datasets/seizures.csv'
#LUNG_CANCER_PATH = './datasets/lung-cancer.data'

# algorithm params
ALGO_PARAMS = {
    'dataset': [
        #{'type': 'holdout', 'problem': 'classification', 'name': "lung_cancer", 'data': BreastDataSet(BREAST_CANCER_PATH)},
        #{'type': 'train_test', 'problem': 'classification', 'name': "adult", 'data': AdultDataSet(ADULT_TRAIN_PATH, ADULT_TEST_PATH)},
        {'type': 'train_test', 'problem': 'classification', 'name': "amazon", 'data': AmazonDataSet(AMAZON_TRAIN_PATH, AMAZON_TEST_PATH)},
    ],

    'decision_function': [
        {'regression': Lasso(), 'classification': LogisticRegression(multi_class='auto', solver='liblinear', penalty='l1', C=1000), 'type': 'linear'},
#        {'regression': MLPRegressor(hidden_layer_sizes=(5, 5), activation='relu'),
#         'classification': MLPClassifier(hidden_layer_sizes=(5, 5), activation='relu')},
#        {'regression': LGBMRegressor(
#                boosting_type='rf',
#                learning_rate=1e-2,
#                max_depth=5,
#                subsample=0.7,
#                n_estimators=200,
#                verbose=-1,
#                subsample_freq=5,
#                num_leaves=2**5,
#                silent=True
#            ),
#        'classification': LGBMClassifier(
#                boosting_type='rf',
#                learning_rate=1e-2,
#                max_depth=5,
#                subsample=0.7,
#                n_estimators=200,
#                verbose=-1,
#                subsample_freq=5,
#                num_leaves=2 ** 5,
#                silent=True
#            )
#        }
    ]
}


HYPERPARAMS = {
    'bins': [2, 4, 8],
    'lambda': [.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
}


def benchmark_holdout(dataset, decision_function, lambda_param, bins):
    dataset['data'].load_from_folder()
    dataset['data'].process_features()

    X_plain = dataset['data'].get_features(False)
    X_transformed = dataset['data'].get_features(True)
    y = dataset['data'].get_target()

    if bins >= y.size * feature_selection_share + 1:
        print(key, bins, 'very small dataset for such dichtomization.')
        raise DichtomizationImpossible(bins, int(y.size * feature_selection_share))

    dfunc = decision_function['classification']
    score_function = partial(log_likelihood_regularized_score_val, _lambda=lambda_param)    #score_function = partial(decision_function, _lambda=lambda_param)

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
        n_jobs=1
    )

    return bench.benchmark(X_plain, X_transformed, y)


def benchmark_train_test(dataset, decision_function, lambda_param, bins):
    dataset['data'].load_train_from_file()
    dataset['data'].load_test_from_file()
    dataset['data'].process_features()

    X_train_plain = dataset['data'].get_train_features(False)
    X_train_transformed = dataset['data'].get_train_features(True)

    X_test_plain = dataset['data'].get_test_features(False)
    X_test_transformed = dataset['data'].get_test_features(True)

    y_train = dataset['data'].get_train_target()
    y_test = dataset['data'].get_test_target()

    if bins >= y_train.size * feature_selection_share + 1:
        print(key, bins, 'very small dataset for such dichtomization.')
        raise DichtomizationImpossible(bins, int(y_train.size * feature_selection_share))

    dfunc = decision_function['classification']
    score_function = partial(log_likelihood_regularized_score_val,
                             _lambda=lambda_param)

    bench = DichtomizedTrainTestBenchmark(
        optimizer=ForwardFeatureSelectionExtended(
            decision_function=dfunc,
            score_function=score_function,
            n_bins=bins,
            train_share=0.8,
            n_cv_ffs=8,
        ),
        decision_function=dfunc,
        n_bins=bins,
    )

    bench.benchmark(X_train_plain, X_train_transformed, y_train, X_test_plain, X_test_transformed, y_test)


if __name__ == '__main__':
    np.random.seed(42)

    feature_selection_share = .5

    joblib.dump('test', "./data/testdoc.bin")

    results = {}

    for dataset, decision_function in product(ALGO_PARAMS['dataset'], ALGO_PARAMS['decision_function']):
        dfunc = decision_function[dataset['problem']]
        key = "{}, {}".format(dataset['name'], dfunc.__class__.__name__)
        results[key] = list()

        print('>>', key)

        for lambda_param, bins in product(HYPERPARAMS['lambda'], HYPERPARAMS['bins']):
            predictions = None
            try:
                if dataset['type'] == 'holdout':
                    predictions = benchmark_holdout(dataset, decision_function, lambda_param, bins)
                elif dataset['type'] == 'train_test':
                    predictions = benchmark_train_test(dataset, decision_function, lambda_param, bins)
                else:
                    print('unknown target type')
            except DichtomizationImpossible as e:
                print(str(e))
                continue

            results[key].append({
                'bins': bins,
                'lambda': lambda_param,
                'result': predictions
            })

            joblib.dump(results, "./data/mlrank_realdata_usm_lik_full_5.bin")
