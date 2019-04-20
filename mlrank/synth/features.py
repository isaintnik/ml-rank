import numpy as np


def shuffle_features(dataset):
    n_features = dataset.shape[1]
    feature_permutation = np.random.choice(n_features, n_features, replace=False)

    return dataset[:, feature_permutation]


class FeaturesGenerator(object):
    @staticmethod
    def generate_normal_features(
            mean_spread: tuple, var_coef: float, n_features: int, n_samples: int
    ) -> np.array:
        means = np.random.uniform(*mean_spread, n_features)
        variances = np.random.poisson(var_coef, n_features) + 1

        features = list()

        for mean, var in np.vstack([means, variances]).T:
            features.append(np.random.normal(mean, var, n_samples))

        return {
            'features': np.vstack(features).T,
            'params': dict(zip(means.tolist(), variances.tolist()))
        }

    @staticmethod
    def generate_uniform_features(
            left_spread: tuple, right_spread: tuple, n_features: int, n_samples: int
    ) -> np.array:
        lefts = np.random.uniform(*left_spread, n_features)
        rights = np.random.uniform(*right_spread, n_features)

        features = list()
        params = list()

        for left, right in np.vstack([lefts, rights]).T:
            _left = np.min([left, right])
            _right = np.max([left, right])

            features.append(np.random.uniform(_left, _right, n_samples))
            params.append((_left, _right))

        return {
            'features': np.vstack(features).T,
            'params': params
        }