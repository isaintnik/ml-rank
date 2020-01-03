import numpy as np
import os
import sys
import warnings

from mlrank.submodular.metrics import mutual_information_regularized_score_penalized
from sklearn.metrics import mutual_info_score

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

from itertools import product

# submodular optimizer
from mlrank.submodular.optimization.multilinear import (
    MultilinearUSMExtended,
    MultilinearUSMClassic
)

# sklearn stuff
from sklearn.utils._joblib import Parallel, delayed
from sklearn.externals import joblib
from sklearn import clone

# models
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier

# problems
from mlrank.synth.linear import LinearProblemGenerator
from mlrank.synth.nonlinear import NonlinearProblemGenerator

from functools import partial


# algorithm params
ALGO_PARAMS = {
    'size': [50, 100, 300, 500],
    'problem': [
        {'name': 'norm_norm', 'generator': partial(LinearProblemGenerator.make_normal_normal, coefs=np.array([2, 5, -3]), n_junk=4)},
        {'name': 'norm_uni', 'generator': partial(LinearProblemGenerator.make_normal_uniform, coefs=np.array([2, 5, -3]), n_junk=4)},
        {'name': 'mc', 'generator': partial(LinearProblemGenerator.make_mc_uniform, coefs=np.array([2, 5, -3]), n_correlated=2, n_junk=4)},

        {'name': 'lc_log', 'generator': partial(NonlinearProblemGenerator.make_nonlinear_linear_combination_problem, coefs=np.array([2, 5, -3]), n_junk=2, func=np.log)},
        {'name': 'r_log_eye', 'generator': partial(NonlinearProblemGenerator.make_nonlinear_relations_problem, coefs=np.array([2, 5, -3]), n_junk=2, functions=[np.log, lambda x: x, np.log])},
        {'name': 'xor', 'generator': partial(NonlinearProblemGenerator.make_xor_continuous_problem, n_ground=5, n_binary_xoring=2, n_junk=2)},
    ],

    'decision_function': [
        LogisticRegression(multi_class='auto', solver='liblinear', penalty='l1', C=1),
        MLPClassifier(hidden_layer_sizes=(5, 5), activation='relu'),
        LGBMClassifier(
            boosting_type='gbdt',
            learning_rate=0.05,
            num_iterations=600,
            max_depth=5,
            n_estimators=600,
            verbose=-1,
            num_leaves=2 ** 5,
            silent=True
        )
    ]
}

# hyperparameters
HYPERPARAMS = {
    'bins': [2, 4, 8],
    'lambda': [0, .3, .6, 1.]
}


def evaluate_model_info_loss(X, y, decision_function, bins, lambda_param, mask):
    score_function = partial(mutual_information_regularized_score_penalized, _lambda=lambda_param, _gamma=0.1)

    ums = MultilinearUSMExtended(
        decision_function,
        score_function,
        n_bins=bins,
        me_eps=.15,
        threshold=.5,
    )

    result = ums.select(X, y)

    return {
        'bins': bins,
        'lambda': lambda_param,
        'result': result,
        'ground': mask
    }


def evaluate_model_generic_loss(X, y, decision_function, bins, lambda_param, mask):
    ums = MultilinearUSMClassic(
        decision_function,
        mutual_info_score,
        n_bins=bins,
        me_eps=.15,
        threshold=.5,
        lambda_param=lambda_param,
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

    #if os.path.isfile("./data/mlrank_stat_lin_nonlin.bin"):
    #    result = joblib.load("./data/mlrank_stat_lin_nonlin.bin")

    for size, problem, decision_function in product(
            ALGO_PARAMS['size'], ALGO_PARAMS['problem'], ALGO_PARAMS['decision_function']
    ):
        key = "{}_{}_{}".format(
            size, problem['name'], decision_function.__class__.__name__
        )

        print(key)

        data = problem['generator'](size)

        y = data['target']
        X = np.hstack(data['features'])
        mask = data['mask']

        results[key] = Parallel(n_jobs=10)(
            delayed(evaluate_model_info_loss)(X, y, clone(decision_function), bins, lambda_param, mask)
            for bins, lambda_param in product(HYPERPARAMS['bins'], HYPERPARAMS['lambda'])
        )

        joblib.dump(results, "./data/mlrank_synth.bin")
