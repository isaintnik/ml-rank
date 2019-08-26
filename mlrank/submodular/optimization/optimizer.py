from mlrank.preprocessing.dichtomizer import (
    dichtomize_vector,
    dichtomize_matrix,
    DichtomizationIssue
)


class SubmodularOptimizer(object):
    """
    Submodular optimizer for discrete target
    """

    def __init__(self):
        pass

    def dichtomize_features(self, X, n_bins):
        return dichtomize_matrix(X, n_bins=n_bins, ordered=False)

    def dichtomize_target(self, y, n_bins):
        return dichtomize_vector(y, n_bins=n_bins, ordered=False)

    def select(self, X, y) -> list:
        raise NotImplementedError()
