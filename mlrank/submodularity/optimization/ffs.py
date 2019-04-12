import numpy as np

from sklearn.base import clone

from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from mlrank.synth.linear import LinearProblemGenerator
from mlrank.preprocessing.dichtomizer import MaxentropyMedianDichtomizationTransformer, dichtomize_matrix
from mlrank.submodularity.functions.metrics_prediction import mutual_information_normalized
from mlrank.submodularity.functions.metrics_dataset import informational_regularization_2


class ForwardFeatureSelection(object):
    def __init__(self, n_bins, lambda_): #n_holdouts, test_share
        self.lambda_ = lambda_
        #self.n_holdouts = n_holdouts
        #self.test_share = test_share
        self.n_bins = n_bins

    def select(self, X_d, X_c, y, n_features, decision_function, extra_loss=False):
        """
        :param X_d: dichtomized features
        :param X_c: continious features
        :param y: regression target
        :param n_features: number of features in optimal subset
        :return: list of length n_features containing indices
        """

        subset = list()

        while len(subset) != n_features:
            max_score = -np.inf
            max_index = -np.inf

            for i in range(X_d.shape[1]):
                if i in subset:
                    continue

                # holdout validation
                loss_mi = mutual_information_normalized(
                    features=X_d[:, subset + [i]],
                    target=y,
                    decision_function=decision_function,
                    n_bins=4
                )

                if extra_loss:
                    subset_entropy = informational_regularization_2(
                        subset + [i], X_d, X_c, decision_function=decision_function, n_bins=self.n_bins
                    )
                else:
                    subset_entropy = 0

                loss = loss_mi - self.lambda_ * subset_entropy

                if loss > max_score:
                    max_score = loss
                    max_index = i

            subset.append(max_index)

        return subset
