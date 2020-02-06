import os
import sys
import warnings
import time
from functools import partial

from mlrank.preprocessing.dichotomizer import DichotomizationImpossible
from mlrank.submodular.metrics import log_likelihood_regularized_score_multiplicative_balanced
from mlrank.submodular.optimization.ffs_parallel import ForwardFeatureSelectionCompositeClient

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

import numpy as np
from sklearn.externals import joblib
from itertools import product
from mlrank.benchmarks.holdout_bench import HoldoutBenchmark
from mlrank.benchmarks.traintest_bench import TrainTestBenchmark


from config import (
    ALGO_PARAMS,
    HYPERPARAMS
)


def benchmark_holdout(
        service_eval_host: str,
        service_eval_port: str,
        service_store_host: str,
        service_store_port: str,

        dataset,
        decision_function,
        lambda_param,
        bins,
    ):
    dataset['data'].load_from_folder()
    dataset['data'].process_features()
    dataset['data'].cache_features()

    if bins >= dataset['data'].get_target().size * 0.8 + 1:
        print(key, bins, 'very small dataset for such dichtomization.')
        raise DichotomizationImpossible(bins, int(dataset['data'].get_target().size * 0.8))

    dfunc = decision_function['classification']

    score_function = partial(log_likelihood_regularized_score_multiplicative_balanced, _lambda=lambda_param)

    bench = HoldoutBenchmark(
        ForwardFeatureSelectionCompositeClient(
            server_clc=service_eval_host,
            port_clc=service_eval_port,

            decision_function=decision_function['type'],
            score_function=score_function,
            n_bins=bins,
            train_share=0.9,
            n_cv_ffs=8,
        ),
        decision_function=dfunc,
        requires_linearisation=decision_function['type'] != 'gbdt',
        n_holdouts=100,
        n_jobs=24
    )

    return bench.benchmark(dataset['data'])


def benchmark_train_test(
        service_eval_host: str,
        service_eval_port: str,

        dataset,
        decision_function,
        lambda_param,
        bins,
    ):
    dataset['data'].load_train_from_file()
    dataset['data'].load_test_from_file()
    dataset['data'].process_features()
    dataset['data'].cache_features()

    y_train = dataset['data'].get_train_target()

    if bins >= y_train.size * 0.8 + 1:
        print(key, bins, 'very small dataset for such dichtomization.')
        raise DichotomizationImpossible(bins, int(y_train.size * 0.8))

    score_function = partial(log_likelihood_regularized_score_multiplicative_balanced, _lambda=lambda_param)

    bench = TrainTestBenchmark(
        optimizer=ForwardFeatureSelectionCompositeClient(
            server_clc=service_eval_host,
            port_clc=service_eval_port,

            decision_function=decision_function['type'],
            score_function=score_function,
            n_bins=bins,
            train_share=0.9,
            n_cv_ffs=8,
        ),
        decision_function=decision_function['classification'],
        requires_linearisation=decision_function['type'] != 'gbdt'
    )

    start_time = time.time()
    result = bench.benchmark(dataset['data'])
    print("--- %s seconds ---" % (time.time() - start_time))

    return result


EVAL_SERVICE_HOST = '35.193.250.1'
EVAL_SERVICE_PORT = '5001'

if __name__ == '__main__':
    np.random.seed(42)

    joblib.dump('test', "./data/testdoc.bin")

    results = {}

    for dataset, decision_function in product([ALGO_PARAMS['dataset'][1]], ALGO_PARAMS['decision_function']):
        dfunc = decision_function[dataset['problem']]
        key = "{}, {}".format(dataset['name'], dfunc.__class__.__name__)
        results[key] = list()

        print('>>', key)

        for lambda_param, bins in product(HYPERPARAMS['lambda'], HYPERPARAMS['bins']):
            print('>> >>', lambda_param, bins)

            if decision_function['type'] not in dataset['supported']:
                continue

            predictions = None
            try:
                if dataset['type'] == 'holdout':
                    predictions = benchmark_holdout(dataset, decision_function, lambda_param, bins)
                elif dataset['type'] == 'train_test':
                    predictions = benchmark_train_test(
                        EVAL_SERVICE_HOST,
                        EVAL_SERVICE_PORT,
                        dataset,
                        decision_function,
                        lambda_param,
                        bins
                    )
                else:
                    print('unknown target type')
            except DichotomizationImpossible as e:
                print(str(e))
                continue

            results[key].append({
                'bins': bins,
                'lambda': lambda_param,
                'result': predictions
            })

            joblib.dump(results, f"./data/{dataset['name']}_composite_gbdt_1.bin")
