import numpy as np
from functools import partial

from sklearn.linear_model import LinearRegression

from mlrank.synth.linear import LinearProblemGenerator
from mlrank.submodularity.metrics.target import mutual_information
from mlrank.submodularity.metrics.subset import informational_regularization_regression

problems = [
    {
        'name': 'multicollinear',
        'problem': LinearProblemGenerator.make_mc_uniform(300, np.array([0.2, 5, 9]), 5, 2),
        'config': [3, 2, 5]
    },
    {
        'name': 'normal_junk_and_residuals',
        'problem': LinearProblemGenerator.make_normal_normal(300, np.array([0.2, 5, 9]), 5),
        'config': [3, 5]
    },
    {
        'name': 'uniform_junk_and_residuals',
        'problem': LinearProblemGenerator.make_normal_uniform(300, np.array([0.2, 5, 9]), 5),
        'config': [3, 5]
    },
]


def get_loss_values(subset, X, y, decision_function, n_bins):
    mi = mutual_information(subset, X, y, decision_function, n_bins=n_bins)
    r = informational_regularization_regression(subset, X, decision_function, n_bins=i)

    return {'mi': mi, 'r': r}


if __name__ == '__main__':
    np.random.seed(42)

    for problem in problems[1:]:
        print('#'*50)
        print(problem['name'])
        print('#'*50)


        y = problem['problem']['target']
        X = np.hstack(problem['problem']['features'])
        config = problem['config']

        decision_function = LinearRegression()

        _lambda = 1

        for i in [4, 8, 16]:
            print('-'*50)
            print('dictomization: ', i)

            get_lv = partial(get_loss_values, X=X, y=y, decision_function=decision_function, n_bins=i)

            a = list(range(config[0]))
            b = list(range(config[0], X.shape[1]))
            c = list(range(0, X.shape[1]))

            a_vals = get_lv(a)
            b_vals = get_lv(b)
            c_vals = get_lv(c)

            print('config (ground): ', a)
            print('metric: ', a_vals['mi'], a_vals['r'], a_vals['mi'] + _lambda * a_vals['r'])
            print('config (noise): ', b)
            print('metric: ', b_vals['mi'], b_vals['r'], b_vals['mi'] + _lambda * b_vals['r'])
            print('config (full): ', c)
            print('metric: ', c_vals['mi'], c_vals['r'], c_vals['mi'] + _lambda * c_vals['r'])
