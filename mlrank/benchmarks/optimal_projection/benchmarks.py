from .optimal_projection_benchmark import OptimalProjectionBenchmark
from .optimal_projection_algorithm import *
from mlrank.hyperparams_opt import (
    get_optimized_logistic_regression,
    get_optimized_svc,
    get_optimized_lightgbm
)

from sklearn.model_selection import train_test_split


def calc_benchmarks(model, X, y, n_min_projections, n_max_projections, n_holdout_validations):
    pca_bench = OptimalProjectionBenchmark(
        model, PCAWrap(), n_holdout_validations=n_holdout_validations,
        n_min_projections=n_min_projections, n_max_projections=n_max_projections
    )

    ica_bench = OptimalProjectionBenchmark(
        model, ICAWrap(), n_holdout_validations=n_holdout_validations,
        n_min_projections=n_min_projections, n_max_projections=n_max_projections
    )

    tsne_bench = OptimalProjectionBenchmark(
        model, TSNEWrap(), n_holdout_validations=n_holdout_validations,
        n_min_projections=n_min_projections, n_max_projections=n_max_projections
    )

    lle_bench = OptimalProjectionBenchmark(
        model, LLEWrap(), n_holdout_validations=n_holdout_validations,
        n_min_projections=n_min_projections, n_max_projections=n_max_projections
    )

    return {
        'pca': pca_bench.benchmark(X, y),
        'ica': ica_bench.benchmark(X, y),
        #'tsne': tsne_bench.benchmark(X, y),
        'lle': lle_bench.benchmark(X, y)
    }


def lr_projection_benchmark(X, y, n_min_projections, n_max_projections, n_holdout_validations):
    X_hyper, X_rest, y_hyper, y_rest = train_test_split(X, y, random_state=42, test_size=.80)
    model = get_optimized_logistic_regression(X_hyper, y_hyper)
    return calc_benchmarks(model, X_rest, y_rest, n_min_projections, n_max_projections, n_holdout_validations)


def svc_projection_benchmark(X, y, n_min_projections, n_max_projections, n_holdout_validations):
    X_hyper, X_rest, y_hyper, y_rest = train_test_split(X, y, random_state=42, test_size=.80)
    model = get_optimized_svc(X_hyper, y_hyper)
    return calc_benchmarks(model, X_rest, y_rest, n_min_projections, n_max_projections, n_holdout_validations)


def rf_projection_benchmark(X, y, n_min_projections, n_max_projections, n_holdout_validations):
    X_hyper, X_rest, y_hyper, y_rest = train_test_split(X, y, random_state=42, test_size=.80)
    model = get_optimized_lightgbm(X_hyper, y_hyper)
    return calc_benchmarks(model, X_rest, y_rest, n_min_projections, n_max_projections, n_holdout_validations)
