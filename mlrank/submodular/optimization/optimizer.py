import numpy as np

from mlrank.preprocessing.dichtomizer import (
    dichotomize_vector,
    dichotomize_matrix,
)

from mlrank.datasets.dataset import DataSet


class SubmodularOptimizer(object):
    """
    Submodular optimizer for discrete target
    """

    def __init__(self):
        pass

    def dichotomize_features(self, X: dict, n_bins, continuous_feature_list: list) -> dict:
        #return dichtomize_matrix(X, n_bins=n_bins, ordered=False)
        new_features = dict()
        for k, v in X.items():
            if k in continuous_feature_list:
                new_features[k] = dichotomize_vector(v, n_bins=n_bins, ordered=False)
            else:
                new_features[k] = v
        return new_features

    def dichotomize_target(self, y: dict, n_bins) -> np.array:
        return dichotomize_vector(y, n_bins=n_bins, ordered=False)

    #def select(self, X_plain: dict, X_transformed:dict,  y: np.array, continuous_feature_list: list) -> list:
    #    raise NotImplementedError()

    def select(self, X_plain: dict, X_transformed: dict, y: np.array, continuous_feature_list: list) -> list:
        raise NotImplementedError()
