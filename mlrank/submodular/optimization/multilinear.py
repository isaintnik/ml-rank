import numpy as np
from sklearn import clone
from sklearn.model_selection import train_test_split

from sklearn.utils._joblib import Parallel, delayed

from mlrank.preprocessing.dichtomizer import (
    DichtomizationIssue
)
from mlrank.submodular.optimization.optimizer import SubmodularOptimizer


from functools import partial

from mlrank.utils import split_dataset


class BaseMultilinearUSM(SubmodularOptimizer):
    def __init__(self,
                 threshold=.5,
                 me_eps=.1,
                 n_jobs=1,
                 n_cv=6,
                 train_share=0.8):
        super().__init__()

        self.threshold = threshold
        self.n_jobs = n_jobs
        self.me_eps = me_eps

        self.n_features = None
        self.feature_list = None
        self.n_cv = n_cv
        self.train_share = train_share

        self.seeds = [(42 + i) for i in range(self.n_cv)]

    def multiliear_extension(self, x) -> np.float128:
        n_iterations = int(1 / (self.me_eps ** 2))

        def make_sample_from_dist(x):
            res = list()
            for i in x:
                res.append(np.argmax(np.random.multinomial(1, [1 - i, i], 1)))
            return np.atleast_1d(np.squeeze(np.argwhere(np.array(res) > 0))).tolist()

        def sample_submodular(loss_func):
            return loss_func(make_sample_from_dist(x))

        # if x is deterministic then return score on deterministic subset
        if len(set(np.unique(x).tolist()).difference([1, 0])) == 0:
            return self.score(np.atleast_1d(np.argwhere(x).squeeze()).tolist())

        x_a = np.array(x)

        # Statistically ', expected_samples, ' out of n_iterations will be sampled. Therefore set all the <1 probabilities as 1
        expected_samples = (1-np.min(x_a[x_a > 0])) * n_iterations

        if expected_samples < 1:
            return self.score(np.atleast_1d(np.argwhere(x_a > 0).squeeze()).tolist())

        if self.n_jobs > 1:
            sampled_losses = Parallel(self.n_jobs)(
                delayed(partial(
                    lambda loss_func: loss_func(make_sample_from_dist(x)),
                    loss_func=self.score)
                )()

                for _ in range(n_iterations)
            )
        else:
            sampled_losses = list()
            for _ in range(n_iterations):
                sampled_losses.append(sample_submodular(self.score))

        return np.mean(sampled_losses)

    def optimize(self):
        x = np.zeros(self.n_features)
        y = np.ones(self.n_features)

        for i in range(self.n_features):
            print(x)
            print(y)

            x_i = np.copy(x)
            x_i[i] = 1

            y_i = np.copy(y)
            y_i[i] = 0

            a_i = self.multiliear_extension(x_i) - self.multiliear_extension(x)
            b_i = self.multiliear_extension(y_i) - self.multiliear_extension(y)

            print(a_i, b_i)

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

            #print(x)

        print(x)
        return np.where(x > self.threshold)[0].tolist()

    def score(self, A):
        raise NotImplementedError()

    def select(self, X_plain: dict, X_transformed: dict, y: np.array, continuous_feature_list: list) -> list:
        raise NotImplementedError()


class MultilinearUSMExtended(BaseMultilinearUSM):
    def __init__(self,
                 decision_function,
                 score_function,
                 n_bins = 4,
                 me_eps = .1,
                 threshold = .5,
                 n_jobs=1,
                 n_cv=6,
                 train_share=0.8
                 ):
        """

        :param decision_function:
        :param n_bins:
        :param me_eps:
        :param lambda_param:
        :param type_of_problem:
        :param n_jobs:
        """
        super().__init__(threshold=threshold, me_eps=me_eps, n_jobs=n_jobs, n_cv=n_cv, train_share=train_share)

        self.n_bins = n_bins
        self.decision_function = clone(decision_function)
        self.score_function = score_function

        self.n_features = None
        self._score_function = None
        self.X_f = None
        self.X_t = None
        self.y = None

        print(me_eps, 'approximation requires', int(1. / (me_eps ** 2)), 'samples from categorical distribution.')

    #def score(self, numeric_features: list):
    #    return self._score_function([self.feature_list[f] for f in numeric_features])

    def score(self, numeric_features: list) -> np.float128:
        scores = list()

        for i in range(self.n_cv):
            result = split_dataset(self.X_t, self.X_f, self.y, self.seeds[i], 1 - self.train_share)

            scores.append(
                self._score_function(
                    A=[self.feature_list[f] for f in numeric_features],
                    X_f=result['train']['transformed'],
                    X_f_test=result['test']['transformed'],
                    X_t=result['train']['plain'],
                    X_t_test=result['test']['plain'],
                    y=result['train']['target'],
                    y_test=result['test']['target'],
                )
            )

        return np.mean(scores)

    def select(self, X_plain: dict, X_transformed: dict, y: np.array, continuous_feature_list: list) -> list:
        self.feature_list = list(X_plain.keys())
        self.n_features = len(X_plain.keys())

        try:
            self.X_f = X_transformed
            self.X_t = self.dichtomize_features(X_plain, self.n_bins, continuous_feature_list)
            self.y = self.dichtomize_target(y, self.n_bins)
        except Exception as e:
            print(e)
            raise DichtomizationIssue(self.n_bins)

        self._score_function = partial(
            self.score_function,
            decision_function=self.decision_function
        )

        return [self.feature_list[i] for i in self.optimize()]
