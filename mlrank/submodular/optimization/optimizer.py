import numpy as np

from mlrank.preprocessing.dichtomizer import (
    dichtomize_vector,
    dichtomize_matrix,
)

from mlrank.datasets.dataset import DataSet


class SubmodularOptimizer(object):
    """
    Submodular optimizer for discrete target
    """

    def __init__(self):
        pass

    def dichtomize_features(self, X: dict, n_bins, continuous_feature_list: list) -> dict:
        #return dichtomize_matrix(X, n_bins=n_bins, ordered=False)
        new_features = dict()
        print('-' * 100)
        print('feature dichtomization')
        print('-' * 100)
        for k, v in X.items():
            if k in continuous_feature_list:
                print(k)
                new_features[k] = dichtomize_vector(v, n_bins=n_bins, ordered=False)
            else:
                new_features[k] = v
        print('-'*100)
        return new_features

    def dichtomize_target(self, y: dict, n_bins) -> np.array:
        return dichtomize_vector(y, n_bins=n_bins, ordered=False)

    #def select(self, X_plain: dict, X_transformed:dict,  y: np.array, continuous_feature_list: list) -> list:
    #    raise NotImplementedError()

    def select(self, X_plain: dict, X_transformed: dict, y: np.array, continuous_feature_list: list) -> list:
        raise NotImplementedError()
