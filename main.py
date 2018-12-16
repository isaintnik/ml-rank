import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from mlrank.benchmarks.optimal_subset import OptimalSubsetBenchmarkFacade
from mlrank.benchmarks.optimal_projection import OptimalProjectionBenchmarkFacade

import pandas as pd
import numpy as np

import os
import argparse


def load_data():
    df = pd.read_csv('./datasets/cancer/breast_cancer.csv')
    y = df.diagnosis.replace('M', 0).replace('B', 1).values
    X = np.asarray(df.drop(['diagnosis', 'id', 'Unnamed: 32'], axis=1).as_matrix())

    X = StandardScaler().fit_transform(X)

    return X, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="benchmarks for various algorithms")
    parser.add_argument('--bench_type', dest='bench_type', default='projections', help='projections/subsets')

    args = parser.parse_args()

    if args.bench_type == 'subsets':
        facade = OptimalSubsetBenchmarkFacade(5, 1, os.path.dirname(os.path.realpath(__file__)) + '/stats')
        facade.build(*load_data())

    if args.bench_type == 'projections':
        facade = OptimalProjectionBenchmarkFacade(1, 10, 50, os.path.dirname(os.path.realpath(__file__)) + '/stats')
        facade.build(*load_data())
