import numpy as np
from .features import FeaturesGenerator


class LinearProblemGenerator(object):
    @staticmethod
    def make_problem_normal_normal(n_samples: int, coefs: np.array, n_junk: int):
        n_ground = coefs.shape[0]

        data = FeaturesGenerator.generate_normal_features(
            mean_spread = (-5, 5),
            var_coef = 2,
            n_features = n_ground,
            n_samples = n_samples
        )

        X_ground, params_ground = data['features'], data['params']

        y = (X_ground * coefs.reshape(1, -1)).sum(1) + np.random.normal(0, 10, n_samples)
            # add noisy features

        data = FeaturesGenerator.generate_normal_features(
            mean_spread=(-5, 5),
            var_coef=2,
            n_features=n_junk,
            n_samples=n_samples
        )

        X_junk, params_junk = data['features'], data['params']

        return (y, X_ground, X_junk)

    @staticmethod
    def make_normal_uniform(n_samples: int, coefs: np.array, n_junk: int):
        n_ground = coefs.shape[0]

        data = FeaturesGenerator.generate_normal_features(
            mean_spread=(-20, 20),
            var_coef=2,
            n_features=n_ground,
            n_samples=n_samples
        )

        X_ground, params_ground = data['features'], data['params']

        y = (X_ground * coefs.reshape(1, -1)).sum(1) + np.random.normal(0, 2)

        # add noisy features

        data = FeaturesGenerator.generate_uniform_features(
            left_spread=(-20, 5),
            right_spread=(10, 100),
            n_features=n_junk,
            n_samples=n_samples
        )

        X_junk, params_junk = data['features'], data['params']

        return (y, X_ground, X_junk)


    @staticmethod
    def make_mc_uniform(n_samples: int, coefs: np.array, n_junk: int, n_correlated: int):
        n_ground = coefs.shape[0]

        data = FeaturesGenerator.generate_normal_features(
            mean_spread=(-20, 20),
            var_coef=2,
            n_features=n_ground,
            n_samples=n_samples
        )

        X_ground, params_ground = data['features'], data['params']

        cor_features = []

        for i in range(n_correlated):
            expl_subset = np.random.choice(n_ground, np.random.random_integers(2, n_ground))
            cor_features.append(
                (X_ground[:, expl_subset] * np.random.uniform(-5, 5, len(expl_subset))).sum(axis=1).reshape(-1, 1)
            )

        X_corr = np.concatenate(cor_features, axis=1)

        y = (X_ground * coefs.reshape(1, -1)).sum(1) + np.random.normal(0, 10)

        # add noisy features

        data = FeaturesGenerator.generate_uniform_features(
            left_spread=(-20, 5),
            right_spread=(10, 100),
            n_features=n_junk,
            n_samples=n_samples
        )

        X_junk, params_junk = data['features'], data['params']

        return (y, X_ground, X_junk, X_corr)
