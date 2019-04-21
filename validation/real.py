from itertools import product

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils._joblib import Parallel, delayed
from sklearn import clone
from sklearn.externals import joblib

from mlrank.submodularity.optimization.usm import MultilinearUSM

from lightgbm import LGBMClassifier

# algorithm params
ALGO_PARAMS = {
    'decision_function': [
        LinearRegression(),
        # MLPRegressor(hidden_layer_sizes=(5, 5)),
        # LGBMRegressor(
        #         boosting_type='rf',
        #         learning_rate=1e-2,
        #         max_depth=5,
        #         subsample=0.7,
        #         n_estimators=200,
        #         verbose=-1,
        #         subsample_freq=5,
        #         num_leaves=2**5,
        #         silent=True
        #     )
    ]
}

# hyperparameters
HYPERPARAMS = {
    'bins': [4], #, 8, 16],
    'lambda': [.5]#[0, .3, .6, 1.]
}


def evaluate_model(X, y, decision_function, bins, lambda_param):
    ums = MultilinearUSM(
        decision_function=decision_function, n_bins=bins, me_eps=.1,
        lambda_param=lambda_param, type_of_problem='classification',
        n_jobs=6
    )
    result = ums.select(X, y)
    return {
        'bins': bins,
        'lambda': lambda_param,
        'result': result[0]
    }


def load_data():
    import os

    #df = pd.read_csv('/Users/ppogorelov/Python/github/ml-rank/datasets/cancer/breast_cancer.csv')
    df = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '/datasets/cancer/breast_cancer.csv')

    y = df.diagnosis.replace('M', 0).replace('B', 1).values
    X = np.asarray(df.drop(['diagnosis', 'id', 'Unnamed: 32'], axis=1).values)

    X = StandardScaler().fit_transform(X)

    return X, y.reshape(-1, 1)


if __name__ == '__main__':
    np.random.seed(42)

    results = {}

    X, y = load_data()

    for decision_function in ALGO_PARAMS['decision_function']:
        key = "{}".format(decision_function.__class__.__name__)

        results[key] = list()

        for bins, lambda_param in product(HYPERPARAMS['bins'], HYPERPARAMS['lambda']):
            results[key].append(evaluate_model(X, y, clone(decision_function), bins, lambda_param))
            joblib.dump(results, "./data/mlrank_stat_cancer.bin")
