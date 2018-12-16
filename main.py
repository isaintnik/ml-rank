import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from mlrank.benchmarks.optimal_subset import *

import pandas as pd
import numpy as np


def to_pandas(bench_result: dict):
    pandas_input = {k: {'nobs': v.nobs, 'mean': v.mean, 'variance': v.variance}
                    for k, v in bench_result.items()}

    return pd.DataFrame(pandas_input).T


def compile_results(pandas_results: dict, by: str='mean'):
    pandas_results = [v[by].rename(k) for k, v in pandas_results.items()]
    return pd.concat(pandas_results, axis=1)


def load_data():
    df = pd.read_csv('./datasets/cancer/breast_cancer.csv')
    y = df.diagnosis.replace('M', 0).replace('B', 1).values
    X = np.asarray(df.drop(['diagnosis', 'id', 'Unnamed: 32'], axis=1).as_matrix())

    X = StandardScaler().fit_transform(X)

    return X, y


if __name__ == '__main__':
    #bench_result_lr = lr_benchmark(*load_data(), 5, 1)
    #bench_result_svc = lr_benchmark(*load_data(), 5, 1)
    results = dict()
    print('calculating benchmarks for svc (4 benchmarks)...')
    results['svc'] = to_pandas(svc_benchmark(*load_data(), 5, 50))
    print('calculating benchmarks for lr (4 benchmarks)...')
    results['lr'] = to_pandas(lr_benchmark(*load_data(), 5, 50))
    print('calculating benchmarks for rf (4 benchmarks)...')
    results['rf'] = to_pandas(rf_benchmark(*load_data(), 5, 50))

    # save to csv
    compile_results(results).to_csv('./stats.csv')

