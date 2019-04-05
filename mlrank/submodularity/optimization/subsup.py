import numpy as np

from sklearn.utils import shuffle

from mlrank.submodularity.functions.metrics_prediction import mutual_information
from mlrank.submodularity.functions.metrics_dataset import joint_entropy_score_ica_estimate
from mlrank.submodularity.optimization.ellipsoid import ellipsoid_submodular_minimize

from functools import partial

joint_entropy = joint_entropy_score_ica_estimate


class SubmodularSupermodularOptimization(object):
    def __init__(self, decision_function, n_splits_target = 4):
        self.n_splits_target = n_splits_target
        self.decision_function = decision_function

    def modular_penalty_approx(self, subset, X):
        # https://arxiv.org/pdf/1207.1404.pdf
        # sampling random permutation starting s.t. permuation[0:|subset|] == subset
        permutation = subset + shuffle(set(range(X.shape[1])).difference(subset)).tolist()
        # calculating that values on our submodular function
        submodular_values = [joint_entropy(permutation[:i], X) for i in range(X.shape[1])]
        # calculating difference of subsets at each unique permutation

        # TODO: check
        support = [0]*len(X.shape[1])
        for i in range(X.shape[1]):
            if i == 1:
                support[permutation[i]] = submodular_values[i]
            else:
                support[permutation[i]] = submodular_values[i] - submodular_values[i - 1]

        support = np.array(support)

        indicator = np.zeros(X.shape[1])
        indicator[subset] = 1

        return support @ indicator

    def submodular_loss(self, subset, X, y):
        if not subset:
            return 0

        first_part = mutual_information(
            X[:, subset], y,
            self.decision_function,
            self.n_splits_target
        )

        second_part = self.modular_penalty_approx(subset, X)

        return first_part - second_part

    def lovasz_gradient(self, x, X, y):
        loss_function = partial(self.submodular_loss, X=X, y=y)

        permutataion = np.argsort(x)
        set_ordered = X[permutataion]

        set_function_values = [0] + [loss_function(set_ordered[:, 0:i]) for i in range(X.shape[1])]

        support = [0] * X.shape[1]
        for i in range(X.shape[1]):
            if i == 1:
                support[permutataion[i]] = set_function_values[i]
            else:
                support[permutataion[i]] = set_function_values[i] - set_function_values[i - 1]

        support = np.array(support)

        return support


    def select(self, X, y):

        current_subset = []

