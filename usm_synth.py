import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from itertools import product

from mlrank.synth.linear import LinearProblemGenerator
from mlrank.preprocessing.dichtomizer import dichtomize_matrix
from mlrank.submodularity.optimization.usm import MultilinearUSM

from sklearn.utils._joblib import Parallel, delayed
from sklearn import clone
from sklearn.externals import joblib

from lightgbm import LGBMRegressor

# algorithm params
ALGO_PARAMS = {
    'size': [40, 100, 200, 500],
    'config': [(3, 5, 2), (3, 2, 5), (5, 5 ,5)],
    'decision_function': [LinearRegression(), LGBMRegressor()]
}

# hyperparameters
HYPERPARAMS = {
    'bins': [4, 8, 16],
    'lambda': [.1, .3, .6, 1.]
}


def evaluate_model(X, y, decision_function, bins, lambda_param, gound):
    ums = MultilinearUSM(
        decision_function,
        bins,
        me_eps=.1,
        lambda_param=lambda_param,
        type_of_problem='regression'
    )

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

    for size, config, decision_function in product(
            ALGO_PARAMS['size'], ALGO_PARAMS['config'], ALGO_PARAMS['decision_function']
    ):
        key = "{}_{}_{}".format(
            size, '_'.join([str(i) for i in config]), decision_function.__class__.__name__
        )

        y, ground, noise, corr = LinearProblemGenerator.make_correlated_uniform(size, *config)#(500, 10, 10, 5)

        X = np.hstack([ground, noise, corr])

        n_ground = ground.shape[1]
        n_noise = noise.shape[1]
        n_corr = corr.shape[1]

        gound = ([1] * n_ground) + ([0] * n_noise) + ([2] * n_corr)

        results[key] = Parallel(n_jobs=6)(
            delayed(evaluate_model)(X, y, clone(decision_function), bins, lambda_param, gound)
            for bins, lambda_param in product(HYPERPARAMS['bins'], HYPERPARAMS['lambda'])
        )

        joblib.dump(results, "./data/mlrank_stat.bin")
