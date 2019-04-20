import numpy as np

from .features import FeaturesGenerator


class NonlinearProblemGenerator(object):
    @staticmethod
    def make_nonlinear_linear_combination_problem(n_samples, coefs: np.array, n_junk: int, func) -> dict:
        n_ground = coefs.shape[0]

        data = FeaturesGenerator.generate_normal_features(
            mean_spread=(30, 60),
            var_coef=10,
            n_features=n_ground,
            n_samples=n_samples
        )

        X_ground, params_ground = data['features'], data['params']

        y = func((X_ground * coefs.reshape(1, -1)).sum(1)) + np.random.normal(0, 1, n_samples)
            # add noisy features

        if n_junk != 0:
            data = FeaturesGenerator.generate_normal_features(
                mean_spread=(-5, 5),
                var_coef=2,
                n_features=n_junk,
                n_samples=n_samples
            )

            X_junk, params_junk = data['features'], data['params']

            return {
                'target':y,
                'features': [X_ground, X_junk]
            }
        else:
            return {
                'target': y,
                'features': [X_ground]
            }

    @staticmethod
    def make_nonlinear_relations_problem(n_samples, coefs: np.array, n_junk: int, functions: list) -> dict:
        n_ground = len(coefs)
        functions_to_features = np.random.random_integers(0, len(functions) - 1, n_ground)

        data = FeaturesGenerator.generate_normal_features(
            mean_spread=(30, 60),
            var_coef=2,
            n_features=n_ground,
            n_samples=n_samples
        )

        X_ground, params_ground = data['features'], data['params']
        X_ground_nonlinear = list()

        for i in range(n_ground):
            X_ground_nonlinear.append(functions[functions_to_features[i]](X_ground[:, i]).reshape(-1, 1))

        X_ground_nonlinear = np.hstack(X_ground_nonlinear)

        y = (X_ground_nonlinear * coefs.reshape(1, -1)).sum(1) + np.random.normal(0, 1, n_samples)

        if n_junk != 0:
            data = FeaturesGenerator.generate_normal_features(
                mean_spread=(-5, 5),
                var_coef=2,
                n_features=n_junk,
                n_samples=n_samples
            )

            X_junk, params_junk = data['features'], data['params']

            return {
                'target': y,
                'features': [X_ground, X_junk]
            }
        else:
            return {
                'target': y,
                'features': [X_ground]
            }

    @staticmethod
    def make_xor_continuous_problem(n_samples, n_ground, n_binary_xoring:int, n_junk: int) -> dict:
        """
        make problem with XOR

        :param n_samples:
        :param n_ground: uniformly distributed variables
        :param n_binary_xoring: bernoulli variables, better take 3 -> 3, 4 -> 6, 5-> 10
        :param n_junk:
        :return:
        """
        assert n_binary_xoring * (n_binary_xoring - 1) / 2 + n_binary_xoring < n_ground

        binary_variables = np.random.random_integers(0, 1, (n_samples, n_binary_xoring))

        data = FeaturesGenerator.generate_normal_features(
            mean_spread=(-5, 5),
            var_coef=2,
            n_features=n_ground,
            n_samples=n_samples
        )

        X_ground_raw, params_ground = data['features'], data['params']
        X_ground_xored = list()

        c_ground = 0

        for i in range(n_binary_xoring):
            for j in range(i, n_binary_xoring):
                X_ground_xored.append((binary_variables[:, i] != binary_variables[:, j]) * X_ground_raw[:, c_ground])
                c_ground += 1

        X_ground_xored = np.hstack(X_ground_xored)
        #X_ground = np.hstack(X_ground_xored + [X_ground_raw[:, c_ground:]] + [binary_variables])

        y = (X_ground_xored + X_ground_raw[:, c_ground:]).sum(1) + np.random.normal(0, 1, n_samples)

        if n_junk != 0:
            data = FeaturesGenerator.generate_normal_features(
                mean_spread=(-5, 5),
                var_coef=2,
                n_features=n_junk,
                n_samples=n_samples
            )

            X_junk, params_junk = data['features'], data['params']

            return {
                'target': y,
                'features': [X_ground_raw, binary_variables, X_junk]
            }
        else:
            return {
                'target': y,
                'features': [X_ground_raw, binary_variables]
            }
