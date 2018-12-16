import pandas as pd
import os

from .benchmarks import (
    svc_optimal_subset_benchmark,
    lr_optimal_subset_benchmark,
    rf_optimal_subset_benchmark
)


class OptimalSubsetBenchmarkFacade(object):
    def __init__(self,
                 n_features,
                 n_holdout_iterations,
                 cache_folder = None
                 ):
        self.n_features = n_features
        self.n_holdout_iterations = n_holdout_iterations
        self.cache_folder = cache_folder

        self.benchmark_results = None

    @staticmethod
    def _to_pandas(bench_results):
        pandas_input = {k: {'nobs': v.nobs, 'mean': v.mean, 'variance': v.variance}
                        for k, v in bench_results.items()}

        return pd.DataFrame(pandas_input).T

    @staticmethod
    def _compile_results(pandas_results: dict, by: str = 'mean'):
        pandas_results = [v[by].rename(k) for k, v in pandas_results.items()]
        return pd.concat(pandas_results, axis=1)

    def cache(self):
        os.makedirs(self.cache_folder, exist_ok=True)
        if self.benchmark_results is None:
            raise Exception('nothing to cache')

        self.benchmark_results.to_csv(os.path.join(self.cache_folder, 'subset.csv'))

    def build(self, X, y):
        results = dict()
        print('calculating benchmarks for svc (4 benchmarks)...')
        results['svc'] = OptimalSubsetBenchmarkFacade._to_pandas(
            svc_optimal_subset_benchmark(X, y, self.n_features, self.n_holdout_iterations))

        print('calculating benchmarks for lr (4 benchmarks)...')
        results['lr'] = OptimalSubsetBenchmarkFacade._to_pandas(
            lr_optimal_subset_benchmark(X, y, self.n_features, self.n_holdout_iterations))

        print('calculating benchmarks for rf (4 benchmarks)...')
        results['rf'] = OptimalSubsetBenchmarkFacade._to_pandas(
            rf_optimal_subset_benchmark(X, y, self.n_features, self.n_holdout_iterations))

        # save to csv
        self.benchmark_results = OptimalSubsetBenchmarkFacade._compile_results(results)
        if self.cache_folder is not None:
            self.cache()

        return self.benchmark_results
