import numpy as np

from sklearn.utils.multiclass import type_of_target

from mlrank.preprocessing.dichtomizer import MaxentropyMedianDichtomizationTransformer, DichtomizationIssue
from mlrank.submodular.optimization.optimizer import SubmodularOptimizer
from .benchmark import Benchmark


class TrainTestBenchmark(Benchmark):
    def __init__(self, optimizer: SubmodularOptimizer, decision_function):
        super().__init__(optimizer, decision_function)

    def evaluate(self, X_train_plain, X_train_transformed, y_train, X_test_plain, X_test_transformed, y_test):
        subset = self.optimizer.select(X_train_plain, X_train_transformed, y_train)
        y_pred = self.train_and_fit(subset, X_train_transformed, y_train, X_test_transformed)

        return {
            'target': np.squeeze(y_test),
            'pred': y_pred,
            'subset': subset
        }

    def benchmark(
            self,
            X_train_plain,
            X_train_transformed,
            y_train,
            X_test_plain=None,
            X_test_transformed=None,
            y_test=None
    ):
        results = [self.evaluate(X_train_plain, X_train_transformed, y_train, X_test_plain, X_test_transformed, y_test)]
        predictions = np.array([(i['pred'], i['target']) for i in results])
        subsets = np.array([i['subset'] for i in results])

        return {
            'predictions': predictions,
            'subsets': subsets
        }


class DichtomizedTrainTestBenchmark(TrainTestBenchmark):
    def __init__(self, optimizer: SubmodularOptimizer, decision_function, n_bins: int):
        super().__init__(optimizer, decision_function)

        self.n_bins = n_bins

    def evaluate(self, X_train_plain, X_train_transformed, y_train, X_test_plain, X_test_transformed, y_test):
        try:
            if type_of_target(y_train) == 'continuous':
                dichtomizer = MaxentropyMedianDichtomizationTransformer(n_splits=self.n_bins)
                dichtomizer.fit(y_train.reshape(-1, 1))

                y_train = dichtomizer.transform(y_train.reshape(-1, 1))
                y_test = dichtomizer.transform(y_test.reshape(-1, 1))

            subset = self.optimizer.select(X_train_plain, X_train_transformed, y_train)
            y_pred = self.train_and_fit(subset, X_train_transformed, y_train, X_test_transformed)

            return {
                'target': np.squeeze(y_test),
                'pred': y_pred,
                'subset': subset
            }
        except DichtomizationIssue as e:
            print(e)
            return dict()