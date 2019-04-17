import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

from itertools import product

from mlrank.synth.linear import LinearProblemGenerator
from mlrank.submodularity.optimization.multilinear_usm import MultilinearUSM

from sklearn.utils._joblib import Parallel, delayed
from sklearn import clone
from sklearn.externals import joblib

from lightgbm import LGBMRegressor

# algorithm params
ALGO_PARAMS = {
    'decision_function': [LinearRegression(), LGBMRegressor()]
}

# hyperparameters
HYPERPARAMS = {
    'bins': [4, 8, 16],
    'lambda': [.1, .3, .6, 1.]
}


def evaluate_model(X, y, decision_function, bins, lambda_param, gound):
    ums = MultilinearUSM(decision_function, bins, me_eps=.1, lambda_param=lambda_param)
    result = ums.select(X, y)
    return {
        'bins': bins,
        'lambda': lambda_param,
        'result': result,
        'ground': gound
    }


if __name__ == '__main__':
    np.random.seed(42)

    results = {}

    for decision_function in ALGO_PARAMS['decision_function']:
        key = "{}".format(decision_function.__class__.__name__)

        results[key] = Parallel(n_jobs=6)(
            delayed(evaluate_model)(X, y, clone(decision_function), bins, lambda_param, gound)
            for bins, lambda_param in product(HYPERPARAMS['bins'], HYPERPARAMS['lambda'])
        )

        joblib.dump(results, "./data/mlrank_stat.bin")
