import numpy as np
from sklearn import clone

from sklearn.utils import shuffle

from sklearn.utils._joblib import Parallel, delayed

from mlrank.preprocessing.dichtomizer import dichtomize_vector, dichtomize_matrix
from mlrank.submodularity.metrics.target import mutual_information_classification
from mlrank.submodularity.metrics.subset import (
    #informational_regularization_regression,
    informational_regularization_classification
)

from functools import partial

from copy import copy

from mlrank.submodularity.optimization.optimizer import SubmodularOptimizer


class MultilinearUSM(SubmodularOptimizer):
    def __init__(self,
                 decision_function,
                 n_bins = 4,
                 me_eps = .1,
                 lambda_param = 1,
                 threshold = .5,
                 n_jobs=1):
        """

        :param decision_function:
        :param n_bins:
        :param me_eps:
        :param lambda_param:
        :param type_of_problem:
        :param n_jobs:
        """
        super().__init__()

        self.n_bins = n_bins
        self.decision_function = clone(decision_function)
        self.threshold = threshold

        self.me_eps = me_eps
        self.lambda_param = lambda_param

        self.n_features = None
        self.metric = None
        self.penalty = None


        self.n_jobs=n_jobs

        self.penalty_raw = informational_regularization_classification

    def dichtomize_features(self, X):
        return dichtomize_matrix(X, n_bins=self.n_bins, ordered=False)

    def dichtomize_target(self, y):
        return dichtomize_vector(y, n_bins=self.n_bins, ordered=False)

    def submodular_loss(self, A):
        if hasattr(A, 'tolist'):
            A = A.tolist()

        if not A:
            return 0

        if self.lambda_param != 0:
            return self.metric(A) + self.lambda_param * self.penalty(A)#self.modular_penalty_approx(A)
        else:
            return self.metric(A)

    def multiliear_extension(self, x):
        def make_sample_from_dist(x):
            res = list()
            for i in x:
                res.append(np.argmax(np.random.multinomial(1, [1 - i, i], 1)))
            return np.atleast_1d(np.squeeze(np.argwhere(np.array(res) > 0)))

        def sample_submodular(loss_func):
            return loss_func(make_sample_from_dist(x))

        if self.n_jobs > 1:
            sampled_losses = Parallel(self.n_jobs)(
                delayed(partial(sample_submodular, loss_func=self.submodular_loss))()
                for _ in range(int(1 / (self.me_eps ** 2)))
            )
        else:
            sampled_losses = list()
            for _ in range(int(1 / (self.me_eps ** 2))):
                sampled_losses.append(sample_submodular(self.submodular_loss))

        return np.mean(sampled_losses)

    def select(self, X, y) -> list:
        X = self.dichtomize_features(X)
        y = self.dichtomize_target(y)

        self.n_features = X.shape[1]

        self.metric = partial(
            mutual_information_classification, X=X, y=y, decision_function=self.decision_function
        )

        self.penalty = partial(self.penalty_raw, X=X, decision_function=self.decision_function)

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

        return np.where(x > self.threshold)[0].tolist()
