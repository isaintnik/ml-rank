import numpy as np
from sklearn import clone
from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

from sklearn.utils._joblib import Parallel, delayed
from sklearn.utils.multiclass import type_of_target

from mlrank.preprocessing.dichtomizer import (
    dichtomize_vector,
    dichtomize_matrix,
    DichtomizationIssue,
    MaxentropyMedianDichtomizationTransformer)
from mlrank.submodular.metrics.target import mutual_information_classification
from mlrank.submodular.optimization.optimizer import SubmodularOptimizer
from mlrank.submodular.metrics.subset import (
    #informational_regularization_regression,
    informational_regularization_classification
)

from functools import partial


class MultilinearUSM(SubmodularOptimizer):
    def __init__(self,
                 threshold=.5,
                 me_eps=.1,
                 n_jobs=1):
        super().__init__()

        self.threshold = threshold
        self.n_jobs = n_jobs
        self.me_eps = me_eps

        self.n_features = None

    def multiliear_extension(self, x) -> float:
        def make_sample_from_dist(x):
            res = list()
            for i in x:
                res.append(np.argmax(np.random.multinomial(1, [1 - i, i], 1)))
            return np.atleast_1d(np.squeeze(np.argwhere(np.array(res) > 0))).tolist()

        def sample_submodular(loss_func):
            return loss_func(make_sample_from_dist(x))

        if self.n_jobs > 1:
            sampled_losses = Parallel(self.n_jobs)(
                delayed(partial(
                    lambda loss_func: loss_func(make_sample_from_dist(x)),
                    loss_func=self.score)
                )()

                for _ in range(int(1 / (self.me_eps ** 2)))
            )
        else:
            sampled_losses = list()
            for _ in range(int(1 / (self.me_eps ** 2))):
                sampled_losses.append(sample_submodular(self.score))

        return float(np.mean(sampled_losses))

    def apply_usm(self):
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

        print(x)
        return np.where(x > self.threshold)[0].tolist()

    def score(self, A):
        raise NotImplementedError()

    def select(self, X, y) -> list:
        raise NotImplementedError()


class MultilinearUSMClassic(MultilinearUSM):
    def __init__(self,
                 decision_function,
                 score_function,
                 n_bins=4,
                 me_eps=.1,
                 lambda_param=1,
                 threshold=.5,
                 n_jobs=1,
                 n_cv=1,
                 train_share=.6):
        """

        :param decision_function:
        :param n_bins:
        :param me_eps:
        :param lambda_param:
        :param type_of_problem:
        :param n_jobs:
        """
        super().__init__(me_eps, n_jobs)

        self.n_bins = n_bins
        self.decision_function = clone(decision_function)
        self.score_function = score_function
        self.threshold = threshold
        self.n_cv = n_cv
        self.train_share = train_share

        #self.me_eps = me_eps
        self.lambda_param = lambda_param

        self.n_features = None
        self._score_function = None

        self.X = None
        self.y = None

        self.seeds = [(42 + i) for i in range(self.n_cv)]
        print(me_eps, 'requires multilinear approximation requires', int(1. / (me_eps ** 2)), 'samples.')

    def score(self, A):
        if not A:
            return 0

        X_s = self.X[:, A]
        y = np.squeeze(self.y)

        scores = list()

        if self.n_cv > 1:
            for i in range(self.n_cv):
                X_train, X_test, y_train, y_test = train_test_split(
                    X_s, y, random_state=self.seeds[i], shuffle=True, test_size=1 - self.train_share
                )

                model = clone(self.decision_function)
                model.fit(X_train, y_train)
                return self.score_function(model.predict(X_test), y_test)

            return float(np.mean(scores))
        else:
            model = clone(self.decision_function)
            model.fit(X_s, y)
            return self.score_function(model.predict(X_s), y)

    def select(self, X, y) -> list:
        self.n_features = X.shape[1]

        self.X = X
        self.y = self.dichtomize_target(y, self.n_bins)
        #self.score = partial(mutual_information_classification, X=X, y=y, decision_function=self.decision_function)
        #self.penalty = partial(informational_regularization_classification, X_f=X, X_t=X_t, decision_function=self.decision_function)

        return self.apply_usm()


class MultilinearUSMExtended(MultilinearUSM):
    def __init__(self,
                 decision_function,
                 score_function,
                 n_bins = 4,
                 me_eps = .1,
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
        self.score_function = score_function
        self.threshold = threshold

        self.me_eps = me_eps

        self.n_features = None
        self._score_function = None
        self.X_f = None
        self.X_t = None
        self.y = None

        self.n_jobs=n_jobs

        print(me_eps, 'requires multilinear approximation requires', int(1. / (me_eps ** 2)), 'samples.')

    def score(self, A):
        return self._score_function(A)

    def select(self, X, y) -> list:
        try:
            self.X_f = X
            self.X_t = self.dichtomize_features(X, self.n_bins)
            self.y = self.dichtomize_target(y, self.n_bins)
        except Exception as e:
            print(e)
            raise DichtomizationIssue(self.n_bins)

        self.n_features = X.shape[1]

        #self.score = partial(mutual_information_classification, X=X, y=y, decision_function=self.decision_function)
        #self.penalty = partial(informational_regularization_classification, X_f=X, X_t=X_t, decision_function=self.decision_function)

        self._score_function = partial(self.score_function, X_f=self.X_f, X_t=self.X_t, y=self.y, decision_function=self.decision_function)

        return self.apply_usm()
