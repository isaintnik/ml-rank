import numpy as np

from sklearn.utils import shuffle

from mlrank.submodularity.functions.metrics_prediction import mutual_information
from mlrank.submodularity.functions.metrics_dataset import informational_regularization_2

from functools import partial

from copy import copy


class MultilinearUSM(object):
    def __init__(self,
                 decision_function,
                 n_bins = 4,
                 me_eps = .1):
                 #algo_eps=.1):
        self.n_bins = n_bins
        self.decision_function = decision_function

        self.me_eps = me_eps
        #self.algo_eps = algo_eps

        self.n_features = None
        self.metric = None
        self.penalty = None

    def modular_penalty_approx(self, subset):
        # https://arxiv.org/pdf/1207.1404.pdf
        # sampling random permutation starting s.t. permuation[0:|subset|] == subset
        permutation = subset + shuffle(list(set(range(self.n_features)).difference(subset)))
        # calculating that values on our submodular function
        submodular_values = [self.penalty(permutation[:i]) for i in range(self.n_features)]
        # calculating difference of subsets at each unique permutation

        # TODO: check
        support = [0]*self.n_features
        for i in range(self.n_features):
            if i == 1:
                support[permutation[i]] = submodular_values[i]
            else:
                support[permutation[i]] = submodular_values[i] - submodular_values[i - 1]

        support = np.array(support)

        indicator = np.zeros(self.n_features)
        indicator[subset] = 1

        return support @ indicator

    def submodular_loss(self, A):
        if hasattr(A, 'tolist'):
            A = A.tolist()

        if not A:
            return 0

        return self.metric(A) + self.penalty(A)#self.modular_penalty_approx(A)

    # def lovasz_gradient(self, x, X, y):
    #     loss_function = partial(self.submodular_loss, X=X, y=y)
    #
    #     permutataion = np.argsort(x)
    #     set_ordered = X[permutataion]
    #
    #     set_function_values = [0] + [loss_function(set_ordered[:, 0:i]) for i in range(X.shape[1])]
    #
    #     support = [0] * X.shape[1]
    #     for i in range(X.shape[1]):
    #         if i == 1:
    #             support[permutataion[i]] = set_function_values[i]
    #         else:
    #             support[permutataion[i]] = set_function_values[i] - set_function_values[i - 1]
    #
    #     support = np.array(support)
    #
    #     return support

    def multiliear_extension(self, x):
        def make_sample_from_dist(x):
            res = list()
            for i in x:
                res.append(np.argmax(np.random.multinomial(1, [1 - i, i], 1)))
            return np.atleast_1d(np.squeeze(np.argwhere(np.array(res) > 0)))

        sampled_losses = list()
        for i in range(int(1 / (self.me_eps ** 2))):
            sample = make_sample_from_dist(x)
            sampled_losses.append(self.submodular_loss(sample))

        return np.mean(sampled_losses)

    def multilinear_grad(self, x):
        zeros_vector = np.zeros(self.n_features)
        ones_vector = np.ones(self.n_features)

        gradient = list()

        for i in range(self.n_features):
            zeros_vector[i] = 1
            ones_vector[i] = 0

            cw_max = np.maximum(x, zeros_vector)
            cw_min = np.minimum(x, ones_vector)

            gradient.append(self.multiliear_extension(cw_max) - self.multiliear_extension(cw_min))

        return np.array(gradient)

    # def pre_process(self):
    #     tau = self.multiliear_extension([1/2] * self.n_features)
    #     gamma = 4 * self.algo_eps * tau
    #     delta = 1/2
    #
    #     mesh = list()
    #     j = 1
    #
    #     while self.algo_eps * j < 1/2:
    #         mesh.append(self.algo_eps * j)
    #         j += 1
    #
    #     mask = list()
    #     for k in mesh:
    #         mask.append(
    #             np.sum(
    #                 self.multilinear_grad([k] * self.n_features) - self.multilinear_grad([1-k] * self.n_features)
    #             ) < 16 * tau
    #         )
    #
    #     if sum(mask) > 0:
    #         delta = np.min(np.array(mesh)[mask])
    #
    #     return {
    #         'x': [delta] * self.n_features,
    #         'y': [1-delta] * self.n_features,
    #         'delta': 1 - 2*delta,
    #         'gamma': gamma
    #     }

    # def update(self, x, y, delta, gamma):
    #     a = self.multilinear_grad(x)
    #     b = -self.multilinear_grad(y)
    #     r = [0] * self.n_features
    #
    #     for u in range(self.n_features):
    #         if a[u] > 0 and b[u] > 0:
    #             r[u] = a[u] / (a[u] + b[u])
    #         elif a[u] > 0:
    #             r[u] = 1

    def select(self, X_d, X_c, y):
        self.n_features = X_c.shape[1]

        self.metric = partial(
            mutual_information, X_c=X_c, y=y, decision_function=self.decision_function, n_bins=self.n_bins
        )

        self.penalty = partial(informational_regularization_2,
            X_d=X_d,
            X_c=X_c,
            decision_function=self.decision_function,
            n_bins=self.n_bins
        )

        x = np.zeros(self.n_features)
        y = np.ones(self.n_features)

        for i in range(self.n_features):
            x_i = copy(x)
            x_i[i] = 1

            y_i = copy(y)
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




