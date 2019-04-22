import numpy as np
import os

import warnings
from sklearn.exceptions import ConvergenceWarning

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

from itertools import product

# submodular optimizer
from mlrank.submodularity.optimization.usm import MultilinearUSM

# sklearn stuff
from sklearn.utils._joblib import Parallel, delayed
from sklearn.externals import joblib
from sklearn import clone

# models
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

# problems
from mlrank.synth.linear import LinearProblemGenerator
from mlrank.synth.nonlinear import NonlinearProblemGenerator

from functools import partial


# algorithm params
ALGO_PARAMS = {
    'size': [50, 100, 200, 500],
    'problem': [
        {'name': 'norm_norm', 'generator': partial(LinearProblemGenerator.make_normal_normal, coefs=np.array([.1, 5, 3]), n_junk=4)},
        {'name': 'norm_uni', 'generator': partial(LinearProblemGenerator.make_normal_uniform, coefs=np.array([.1, 5, 3]), n_junk=4)},
        {'name': 'mc', 'generator': partial(LinearProblemGenerator.make_mc_uniform, coefs=np.array([.1, 5, 3]), n_correlated=2, n_junk=4)},

        {'name': 'lc_log', 'generator': partial(NonlinearProblemGenerator.make_nonlinear_linear_combination_problem, coefs=np.array([.1, 5, 3]), n_junk=2, func=np.log)},
        {'name': 'r_log_eye', 'generator': partial(NonlinearProblemGenerator.make_nonlinear_relations_problem, coefs=np.array([.1, 5, 3]), n_junk=2, functions=[np.log, lambda x: x, np.log])},
        {'name': 'xor', 'generator': partial(NonlinearProblemGenerator.make_xor_continuous_problem, n_ground=5, n_binary_xoring=2, n_junk=2)},
    ],

    'decision_function': [
        LinearRegression(),
        MLPRegressor(hidden_layer_sizes=(5, 5)),
        LGBMRegressor(
                boosting_type='rf',
                learning_rate=1e-2,
                max_depth=5,
                subsample=0.7,
                n_estimators=200,
                verbose=-1,
                subsample_freq=5,
                num_leaves=2**5,
                silent=True
            )
    ]
}

# hyperparameters
HYPERPARAMS = {
    'bins': [4, 8, 16],
    'lambda': [0, .3, .6, 1.]
}


def evaluate_model(X, y, decision_function, bins, lambda_param, mask):
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
        'ground': mask
    }


if __name__ == '__main__':
    np.random.seed(42)

    results = {}

    if os.path.isfile("./data/mlrank_stat_lin_nonlin.bin"):
        result = joblib.load("./data/mlrank_stat_lin_nonlin.bin")

    for size, problem, decision_function in product(
            ALGO_PARAMS['size'], ALGO_PARAMS['problem'], ALGO_PARAMS['decision_function']
    ):
        # //_-
        if size > 50 and not (size == 100 and problem['name'] in ['mc', 'norm_norm', 'norm_uni']):

            key = "{}_{}_{}".format(
                size, problem['name'], decision_function.__class__.__name__
            )

            data = problem['generator'](size)

            y = data['target']
            X = np.hstack(data['features'])
            mask = data['mask']

            results[key] = Parallel(n_jobs=14)(
                delayed(evaluate_model)(X, y, clone(decision_function), bins, lambda_param, mask)
                for bins, lambda_param in product(HYPERPARAMS['bins'], HYPERPARAMS['lambda'])
            )

            joblib.dump(results, "./data/mlrank_stat_lin_nonlin.bin")
