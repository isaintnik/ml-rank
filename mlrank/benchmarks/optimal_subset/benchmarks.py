from .optimal_subset_benchmark import OptimalSubsetBenchmark
from .optimal_subset_alrorithm import *
from mlrank.hyperparams_opt import (
    get_optimized_logistic_regression,
    get_optimized_svc,
    get_optimized_lightgbm
)

#from sklearn.model_selection import train_test_split


def make_mlrank_benchmarks(model, X, y, n_features, n_holdout_validations):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC
    # TODO: kostyli

    if isinstance(model, LinearSVC):
        model = CalibratedClassifierCV(model)

    return {
        'mlr_4b': OptimalSubsetBenchmark(
            MLRankWrap(model, n_features, params={
                'transformation': lambda a, b: a != b,
                'verbose': 1, 'n_splits': 4
            }),
            n_holdout_validations=n_holdout_validations
        ).benchmark(X, y).get_stats(),
        'mlr_8b': OptimalSubsetBenchmark(
            MLRankWrap(model, n_features, params={
                'transformation': lambda a, b: a != b,
                'verbose': 1, 'n_splits': 8
            }),
            n_holdout_validations=n_holdout_validations
        ).benchmark(X, y).get_stats(),
        'mlr_16b': OptimalSubsetBenchmark(
            MLRankWrap(model, n_features, params={
                'transformation': lambda a, b: a != b,
                'verbose': 1, 'n_splits': 16
            }),
            n_holdout_validations=n_holdout_validations
        ).benchmark(X, y).get_stats(),
        'mlr_32b': OptimalSubsetBenchmark(
            MLRankWrap(model, n_features, params={
                'transformation': lambda a, b: a != b,
                'verbose': 1, 'n_splits': 32
            }),
            n_holdout_validations=n_holdout_validations
        ).benchmark(X, y).get_stats()
    }


def calc_benchmarks(model, X, y, n_features, n_holdout_validations):
    sfs_bench = OptimalSubsetBenchmark(SFSWrap(model, n_features), n_holdout_validations=n_holdout_validations)
    rfe_bench = OptimalSubsetBenchmark(RFEWrap(model, n_features), n_holdout_validations=n_holdout_validations)
    lrc_bench = OptimalSubsetBenchmark(LRCoefficentsWrap(model, n_features), n_holdout_validations=n_holdout_validations)
    rff_bench = OptimalSubsetBenchmark(RFImportancesWrap(model, n_features), n_holdout_validations=n_holdout_validations)

    return {
        #'sfs': sfs_bench.benchmark(X, y).get_stats(),
        #'rfe': rfe_bench.benchmark(X, y).get_stats(),
        #'lrc': lrc_bench.benchmark(X, y).get_stats(),
        #'rff': rff_bench.benchmark(X, y).get_stats(),
        **make_mlrank_benchmarks(model, X, y, n_features, n_holdout_validations)
    }


def lr_optimal_subset_benchmark(X, y, n_features, n_holdout_validations):
    #X_hyper, X_rest, y_hyper, y_rest = train_test_split(X, y, random_state=42, test_size=.80)
    model = get_optimized_logistic_regression(X, y)
    return calc_benchmarks(model, X, y, n_features, n_holdout_validations)


def svc_optimal_subset_benchmark(X, y, n_features, n_holdout_validations):
    #X_hyper, X_rest, y_hyper, y_rest = train_test_split(X, y, random_state=42, test_size=.80)
    model = get_optimized_svc(X, y)
    return calc_benchmarks(model, X, y, n_features, n_holdout_validations)


def rf_optimal_subset_benchmark(X, y, n_features, n_holdout_validations):
    #X_hyper, X_rest, y_hyper, y_rest = train_test_split(X, y, random_state=42, test_size=.80)
    model = get_optimized_lightgbm(X, y)
    return calc_benchmarks(model, X, y, n_features, n_holdout_validations)
