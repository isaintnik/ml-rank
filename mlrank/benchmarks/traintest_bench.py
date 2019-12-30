import numpy as np

from sklearn.utils.multiclass import type_of_target

from mlrank.preprocessing.dichtomizer import MaxentropyMedianDichtomizationTransformer, DichtomizationIssue
from mlrank.submodular.optimization.optimizer import SubmodularOptimizer
from .benchmark import Benchmark
from mlrank.datasets.dataset import SeparatedDataset


class TrainTestBenchmark(Benchmark):
    def __init__(self, optimizer: SubmodularOptimizer, decision_function):
        super().__init__(optimizer, decision_function)

    def evaluate(self, dataset: SeparatedDataset):
        subset = self.optimizer.select(
            dataset.get_train_features(False),
            dataset.get_train_features(True),
            dataset.get_train_target(),
            dataset.get_continuous_feature_names()
        )

        y_pred = self.train_and_fit(
            subset,
            dataset.get_train_features(convert_to_linear=True),
            dataset.get_train_target(),
            dataset.get_test_features(convert_to_linear=True)
        )

        print(np.abs((np.squeeze(dataset.get_test_target()) - y_pred)).sum(), y_pred.shape)

        return {
            'target': np.squeeze(dataset.get_test_target()),
            'pred': y_pred,
            'subset': subset
        }

    #def benchmark(
    #        self,
    #        continuous_feature_names,
    #        X_train_plain,
    #        X_train_transformed,
    #        y_train,
    #        X_test_plain=None,
    #        X_test_transformed=None,
    #        y_test=None
    #):
    def benchmark(self, dataset: SeparatedDataset):
        results = [self.evaluate(dataset)]
        predictions = np.array([(i['pred'], i['target']) for i in results])
        subsets = np.array([i['subset'] for i in results])

        return {
            'predictions': predictions,
            'subsets': subsets
        }
