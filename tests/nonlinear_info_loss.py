import numpy as np
from functools import partial

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor

from mlrank.synth.nonlinear import NonlinearProblemGenerator
from mlrank.submodularity.metrics.target import mutual_information
from mlrank.submodularity.metrics.subset import informational_regularization_regression

problems = [
    {
        'name': 'xor',
        'problem': NonlinearProblemGenerator.make_xor_continuous_problem(300, 4, 3, 4)
    },
    {
        'name': 'lc_log',
        'problem': NonlinearProblemGenerator.make_nonlinear_linear_combination_problem(300, np.array([0.2, 5, 9]), 5, np.log)
    },
    {
        'name': 'r_log',
        'problem': NonlinearProblemGenerator.make_nonlinear_relations_problem(1000, np.array([0.2, 5, 9]), 2,  [np.log, lambda x: x])
    }
]


def get_loss_values(subset, X, y, decision_function, n_bins):
    mi = mutual_information(subset, X, y, decision_function, n_bins=n_bins)
    r = informational_regularization_regression(subset, X, decision_function, n_bins=i)

    return {'mi': mi, 'r': r}


if __name__ == '__main__':
    np.random.seed(42)

    for problem in problems:
        print('#'*50)
        print(problem['name'])
        print('#'*50)

        y = problem['problem']['target']
        X = np.hstack(problem['problem']['features'])
        mask = np.array(problem['problem']['mask'])

        decision_function = LinearRegression()#MLPRegressor(hidden_layer_sizes=(3,2), activation='relu', solver='lbfgs')

        _lambda = 1

        for i in [4, 8, 16]:
            print('-'*50)
            print('dictomization: ', i)
            get_lv = partial(get_loss_values, X=X, y=y, decision_function=decision_function, n_bins=i)

            a = np.where(mask)[0].tolist()
            b = np.where(mask == 0)[0].tolist()
            c = np.where((mask == 0) | (mask == 1))[0].tolist()

            a_vals = get_lv(a)
            b_vals = get_lv(b)
            c_vals = get_lv(c)

            print('config (ground) : ', a)
            print('metric: ', a_vals['mi'], a_vals['r'], a_vals['mi'] + _lambda * a_vals['r'])
            print('config (noise): ', b)
            print('metric: ', b_vals['mi'], b_vals['r'], b_vals['mi'] + _lambda * b_vals['r'])
            print('config (full): ', c)
            print('metric: ', c_vals['mi'], c_vals['r'], c_vals['mi'] + _lambda * c_vals['r'])