import numpy as np

from sklearn.utils import shuffle

from sklearn.utils._joblib import Parallel, delayed

from mlrank.submodularity.functions.metrics_prediction import mutual_information
from mlrank.submodularity.functions.metrics_dataset import (
    informational_regularization_regression,
    informational_regularization_classification
)

from functools import partial

from copy import copy


class MultilinearUSM(object):
    def __init__(self,
                 decision_function,
                 n_bins = 4,
                 me_eps = .1,
                 lambda_param = 1,
                 type_of_problem = 'regression',
                 n_jobs=1):
        self.n_bins = n_bins
        self.decision_function = decision_function

        self.me_eps = me_eps
        self.lambda_param = lambda_param

        self.n_features = None
        self.metric = None
        self.penalty = None

        self.n_jobs=n_jobs

        if type_of_problem == 'classification':
            self.penalty_raw = informational_regularization_classification
        else:
            self.penalty_raw = informational_regularization_regression

    def submodular_loss(self, A):
        if hasattr(A, 'tolist'):
            A = A.tolist()

        if not A:
            return 0

        return self.metric(A) + self.lambda_param * self.penalty(A)#self.modular_penalty_approx(A)

    def multiliear_extension(self, x):
        def make_sample_from_dist(x):
            res = list()
            for i in x:
                res.append(np.argmax(np.random.multinomial(1, [1 - i, i], 1)))
            return np.atleast_1d(np.squeeze(np.argwhere(np.array(res) > 0)))

        def sample_submodular(loss_func):
            return loss_func(make_sample_from_dist(x))

        sampled_losses = Parallel(self.n_jobs)(
            delayed(partial(sample_submodular, loss_func=self.submodular_loss))()
            for i in range(int(1 / (self.me_eps ** 2)))
        )

        return np.mean(sampled_losses)

    def select(self, X, y):
        self.n_features = X.shape[1]

        self.metric = partial(
            mutual_information, X=X, y=y, decision_function=self.decision_function, n_bins=self.n_bins
        )

        self.penalty = partial(self.penalty_raw, X=X, decision_function=self.decision_function, n_bins=self.n_bins)

        x = np.zeros(self.n_features)
        y = np.ones(self.n_features)

        for i in range(self.n_features):
            x_i = np.copy(x)
            x_i[i] = 1

            y_i = np.copy(y)
            y_i[i] = 0

            a_i = self.multiliear_extension(x_i) - self.multiliear_extension(x)
            b_i = self.multiliear_extension(y_i) - self.multiliear_extension(y)

            a_i = max(a_i, 0)
            b_i = max(b_i, 0)

            a = np.zeros(self.n_features)
            b = np.zeros(self.n_features)

            if a_i == 0 and b_i == 0:
                a[i] = 1
                b[i] = 0
            else:
                a[i] = a_i / (a_i + b_i)
                b[i] = b_i / (a_i + b_i)

            x = x + a
            y = y - b

        return x, y