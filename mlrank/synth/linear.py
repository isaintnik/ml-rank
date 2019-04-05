import numpy as np


class LinearProblemGenerator(object):
    @staticmethod
    def generate_normal_features(mean_spread: tuple, var_coef: float, n_features: int, n_samples: int) -> np.array:
        means = np.random.uniform(*mean_spread, n_features)
        variances = np.random.poisson(var_coef, n_features) + 1

        features = list()

        for mean, var in np.vstack([means, variances]).T:
            features.append(np.random.normal(mean, var, n_samples))

        return np.vstack(features).T

    @staticmethod
    def generate_uniform_features(left_spread: tuple, right_spread: tuple, n_features: int, n_samples: int) -> np.array:
        lefts = np.random.uniform(*left_spread, n_features)
        rights = np.random.uniform(*right_spread, n_features)

        features = list()

        for left, right in np.vstack([lefts, rights]).T:
            _left = np.min([left, right])
            _right = np.max([left, right])

            features.append(np.random.uniform(_left, _right, n_samples))

        return np.vstack(features).T

    @staticmethod
    def make_problem_normal_normal(n_samples: int, n_discriptive: int, n_junk: int):
        coefs = np.random.uniform(-5, 5, n_discriptive)
        X_discriptive = LinearProblemGenerator.generate_normal_features(
            mean_spread = (-5, 5),
            var_coef = 2,
            n_features = n_discriptive,
            n_samples = n_samples
        )

        y = (X_discriptive * coefs.reshape(1, -1)).sum(1) \
            + np.random.normal(0, 1, n_samples) \
            + np.random.normal(1, 2, n_samples) \
            + np.random.normal(2, 3, n_samples) \
            + np.random.normal(3, 4, n_samples) \
            # add noisy features

        X_junk = LinearProblemGenerator.generate_normal_features(
            mean_spread=(-5, 5),
            var_coef=2,
            n_features=n_junk,
            n_samples=n_samples
        )

        return (y, X_discriptive, X_junk)

    @staticmethod
    def make_normal_uniform(n_samples: int, n_discriptive: int, n_junk: int):
        coefs = np.random.uniform(-5, 5, n_discriptive)

        X_discriptive = LinearProblemGenerator.generate_normal_features(
            mean_spread=(-20, 20),
            var_coef=2,
            n_features=n_discriptive,
            n_samples=n_samples
        )

        y = (X_discriptive * coefs.reshape(1, -1)).sum(1) + np.random.normal(0, 2)

        # add noisy features

        X_junk = LinearProblemGenerator.generate_uniform_features(
            left_spread=(-20, 5),
            right_spread=(10, 100),
            n_features=n_junk,
            n_samples=n_samples
        )

        return (y, X_discriptive, X_junk)


    @staticmethod
    def make_correlated_uniform(n_samples: int, n_discriptive: int, n_junk: int, n_correlated: int):
        coefs = np.random.uniform(-5, 5, n_discriptive)

        X_discriptive = LinearProblemGenerator.generate_normal_features(
            mean_spread=(-20, 20),
            var_coef=2,
            n_features=n_discriptive,
            n_samples=n_samples
        )

        cor_features = []

        for i in range(n_correlated):
            expl_subset = np.random.choice(n_discriptive, np.random.random_integers(2, n_discriptive))
            cor_features.append(
                (X_discriptive[:, expl_subset] * np.random.uniform(-5, 5, len(expl_subset))).sum(axis=1).reshape(-1, 1)
            )

        X_corr = np.concatenate(cor_features, axis=1)

        y = (X_discriptive * coefs.reshape(1, -1)).sum(1) + np.random.normal(0, 10)

        # add noisy features

        X_junk = LinearProblemGenerator.generate_uniform_features(
            left_spread=(-20, 5),
            right_spread=(10, 100),
            n_features=n_junk,
            n_samples=n_samples
        )

        return (y, X_discriptive, X_junk, X_corr)
