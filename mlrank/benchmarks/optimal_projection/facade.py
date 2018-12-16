import pandas as pd
import os

from .benchmarks import (
    svc_projection_benchmark,
    lr_projection_benchmark,
    rf_projection_benchmark
)


class OptimalProjectionBenchmarkFacade(object):
    def __init__(self,
                 n_min_projections,
                 n_max_projections,
                 n_holdout_iterations,
                 cache_folder = None
                 ):
        self.n_min_projections = n_min_projections
        self.n_max_projections = n_max_projections
        self.n_holdout_iterations = n_holdout_iterations
        self.cache_folder = cache_folder

        self.benchmark_results = None

    @staticmethod
    def _to_pandas(bench_results, by='mean'):
        pandas_input = {
            k: {'nobs': [i.nobs for i in v], 'mean': [i.mean for i in v], 'variance': [i.variance for i in v]}
            for k, v in bench_results.items()
        }

        return pd.DataFrame({k: pd.Series(v[by]).rename(k) for k, v in pandas_input.items()})

    def cache(self):
        os.makedirs(self.cache_folder, exist_ok=True)
        if self.benchmark_results is None:
            raise Exception('nothing to cache')

        for model, data in self.benchmark_results.items():
            data.to_csv(os.path.join(self.cache_folder, f'projection_{model}.csv'))

    def build(self, X, y):
        results = dict()
        print('calculating benchmarks for svc (4 benchmarks)...')
        results['svc'] = OptimalProjectionBenchmarkFacade._to_pandas(svc_projection_benchmark(
            X, y, self.n_min_projections, self.n_max_projections, self.n_holdout_iterations
        ))

        print('calculating benchmarks for lr (4 benchmarks)...')
        results['lr'] = OptimalProjectionBenchmarkFacade._to_pandas(lr_projection_benchmark(
            X, y, self.n_min_projections, self.n_max_projections, self.n_holdout_iterations
        ))

        print('calculating benchmarks for rf (4 benchmarks)...')
        results['rf'] = OptimalProjectionBenchmarkFacade._to_pandas(rf_projection_benchmark(
            X, y, self.n_min_projections, self.n_max_projections, self.n_holdout_iterations
        ))

        # save to csv
        self.benchmark_results = results
        if self.cache_folder is not None:
            self.cache()

        return self.benchmark_results
