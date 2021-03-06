import numpy as np

from sklearn import clone

from mlrank.submodular.optimization.optimizer import SubmodularOptimizer
from mlrank.utils import make_features_matrix

from mlrank.datasets.dataset import DataSet


class Benchmark(object):
    def __init__(self,
         optimizer: SubmodularOptimizer,
         decision_function,
         requires_linearisation: bool
    ):
        self.optimizer = optimizer
        self.decision_function = decision_function
        self.requires_linearisation = requires_linearisation

    def train_and_fit(self, subset, X_train: dict, y_train: np.array, X_test: dict):
        if not subset:
            return None

        X_train_df = make_features_matrix(X_train, subset)
        X_test_df = make_features_matrix(X_test, subset)

        model = clone(self.decision_function)
        model.fit(X_train_df, np.squeeze(y_train))

        return model.predict(X_test_df)

    def benchmark(self, dataset: DataSet):
        raise NotImplementedError()
