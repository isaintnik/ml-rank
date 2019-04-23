import numpy as np

from sklearn import clone
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target

from mlrank.preprocessing.dichtomizer import MaxentropyMedianDichtomizationTransformer, map_continious_names
from mlrank.submodularity.optimization.optimizer import SubmodularOptimizer


class HoldoutBenchmark(object):
    def __init__(self, optimizer: SubmodularOptimizer, feature_selection_share: float, decision_function, n_holdouts: int):
        if feature_selection_share >= 1 or feature_selection_share <= 0:
            raise Exception('feature_selection_share should be in range (0, 1).')

        self.optimizer=optimizer
        self.feature_selection_share=feature_selection_share
        self.decision_function=decision_function
        self.n_holdouts = n_holdouts

    def benchmark(self, X, y):
        predictions = list()

        for i in range(self.n_holdouts):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=int(1), random_state=42 + i)
            X_f, X_h, y_f, y_h = train_test_split(X_train, y_train, test_size=1-self.feature_selection_share, random_state=42 + i)

            subset = self.optimizer.select(X_f, y_f,)

            model = clone(self.decision_function)
            model.fit(X_h[:, subset], np.squeeze(y_h))

            predictions.append([model.predict(X_test[:, subset]), y_test])

        return np.array(predictions)
