import numpy as np

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


# algorithm params
ALGO_PARAMS = {
    'size': [50, 100, 200, 500],
    'problems': [(3, 5, 2), (3, 2, 5), (5, 5, 5)],
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
                subsample_freq=5, # ????
                num_leaves=2**5,
                silent=True
            )]
}

# hyperparameters
HYPERPARAMS = {
    'bins': [4, 8, 16],
    'lambda': [0, .3, .6, 1.]
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

        y, ground, noise, corr = LinearProblemGenerator.make_mc_uniform(size, *config)#(500, 10, 10, 5)

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
