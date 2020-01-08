import os
import sys
import warnings
from functools import partial

from mlrank.datasets.internet import InternetDataSet
from mlrank.preprocessing.dichotomizer import DichotomizationImpossible
from mlrank.submodular.metrics import log_likelihood_regularized_score_val, bic_criterion, aic_criterion, aicc_criterion
from mlrank.submodular.optimization import ForwardFeatureSelectionExtended, MultilinearUSMExtended

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

import numpy as np
from sklearn.externals import joblib
from itertools import product
from mlrank.benchmarks.holdout_bench import HoldoutBenchmark
from mlrank.benchmarks.traintest_bench import TrainTestBenchmark
from mlrank.datasets import (AdultDataSet, AmazonDataSet, BreastDataSet)

# models
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.neural_network import MLPRegressor, MLPClassifier

from config import (
    ALGO_PARAMS,
    HYPERPARAMS
)


def benchmark_holdout(dataset, decision_function, bins, criterion):
    dataset['data'].load_from_folder()
    dataset['data'].process_features()
    dataset['data'].cache_features()

    if bins >= dataset['data'].get_target().size * 0.8 + 1:
        print(key, bins, 'very small dataset for such dichtomization.')
        raise DichotomizationImpossible(bins, int(dataset['data'].get_target().size * 0.8))

    dfunc = decision_function['classification']

    bench = HoldoutBenchmark(
        ForwardFeatureSelectionExtended(
            decision_function=dfunc,
            score_function=criterion,
            n_bins=bins,
            train_share=0.9,
            n_cv_ffs=8,
            n_jobs=1
        ),
        #MultilinearUSMExtended(
        #    decision_function=dfunc,
        #    score_function=score_function,
        #    n_bins=bins,
        #    train_share=0.8,
        #    n_cv=8,
        #),
        decision_function=dfunc,
        requires_linearisation=decision_function['type'] != 'gbdt',
        n_holdouts=80,
        n_jobs=24
    )

    return bench.benchmark(dataset['data'])


def benchmark_train_test(dataset, decision_function, criterion, bins):
    dataset['data'].load_train_from_file()
    dataset['data'].load_test_from_file()
    dataset['data'].process_features()
    dataset['data'].cache_features()

    y_train = dataset['data'].get_train_target()

    if bins >= y_train.size * 0.8 + 1:
        print(key, bins, 'very small dataset for such dichtomization.')
        raise DichotomizationImpossible(bins, int(y_train.size * 0.8))

    dfunc = decision_function['classification']

    bench = TrainTestBenchmark(
        optimizer=ForwardFeatureSelectionExtended(
            decision_function=dfunc,
            score_function=criterion,
            n_bins=bins,
            train_share=0.9,
            n_cv_ffs=8,
            n_jobs=8
        ),
        decision_function=dfunc,
        requires_linearisation=decision_function['type'] != 'gbdt'
    )

    return bench.benchmark(dataset['data'])

#ARRHYTHMIA_PATH = './datasets/arrhythmia.data'
#FOREST_FIRE_PATH = './datasets/forestfires.csv'
#HEART_DESEASE_PATH = './datasets/reprocessed.hungarian.data'
#SEIZURES_PATH = './datasets/seizures.csv'
#LUNG_CANCER_PATH = './datasets/lung-cancer.data'


if __name__ == '__main__':
    np.random.seed(42)

    joblib.dump('test', "./data/testdoc.bin")

    results = {}

    for dataset, decision_function in product([ALGO_PARAMS['dataset'][1]], ALGO_PARAMS['decision_function']):
        dfunc = decision_function[dataset['problem']]
        key = "{}, {}".format(dataset['name'], dfunc.__class__.__name__)
        results[key] = list()

        print('>>', key)

        for criterion_name, criterion in {
            'aic': aic_criterion,
            'bic': bic_criterion,
            'aicc': aicc_criterion
        }.items():

            for bins in HYPERPARAMS['bins']:
                print('>> >>', bins)

                if decision_function['type'] not in dataset['supported']:
                    continue

                predictions = None
                try:
                    if dataset['type'] == 'holdout':
                        predictions = benchmark_holdout(
                            dataset, decision_function, criterion=criterion, bins=bins
                        )
                    elif dataset['type'] == 'train_test':
                        predictions = benchmark_train_test(
                            dataset, decision_function, criterion=criterion, bins=bins
                        )
                    else:
                        print('unknown target type')
                except DichotomizationImpossible as e:
                    print(str(e))
                    continue

                results[key].append({
                    'bins': bins,
                    'criterion': criterion_name,
                    'result': predictions
                })

                joblib.dump(results, f"./data/{dataset['name']}_criterion.bin")
