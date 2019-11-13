import numpy as np

from mlrank.preprocessing.dichtomizer import (
    dichtomize_vector,
    dichtomize_matrix,
)


class SubmodularOptimizer(object):
    """
    Submodular optimizer for discrete target
    """

    def __init__(self):
        pass

    def dichtomize_features(self, X: dict, n_bins) -> dict:
        #return dichtomize_matrix(X, n_bins=n_bins, ordered=False)
        new_features = dict()
        for k, v in X.items():
            new_features[k] = dichtomize_vector(v, n_bins=n_bins, ordered=False)
        return new_features

    def dichtomize_target(self, y: dict, n_bins) -> np.array:
        return dichtomize_vector(y, n_bins=n_bins, ordered=False)

    def select(self, X_plain: dict, X_transformed:dict,  y: np.array) -> list:
        raise NotImplementedError()
