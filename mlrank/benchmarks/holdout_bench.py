import numpy as np

from sklearn.externals.joblib import Parallel, delayed

from mlrank.submodular.optimization.optimizer import SubmodularOptimizer
from mlrank.utils import split_dataset
from .benchmark import Benchmark
from mlrank.datasets.dataset import HoldoutDataset


class HoldoutBenchmark(Benchmark):
    def __init__(
            self,
            optimizer: SubmodularOptimizer,
            decision_function,
            requires_linearisation: bool,
            n_holdouts: int,
            n_jobs: int
    ):

        super().__init__(optimizer, decision_function, requires_linearisation)

        self.n_holdouts = n_holdouts
        self.n_jobs = n_jobs

    def evaluate(self, dataset: HoldoutDataset, seed):
        result = split_dataset(
            dataset.get_features(False),
            dataset.get_features(self.requires_linearisation),
            dataset.get_target(),
            seed,
            int(1)
        )

        subset = self.optimizer.select(
            result['train']['plain'],
            result['train']['transformed'],
            result['train']['target'],
            dataset.get_continuous_feature_names()
        )

        y_pred = self.train_and_fit(
            subset,
            result['train']['transformed'],
            result['train']['target'],
            result['test']['transformed']
        )

        return {
            'target': np.squeeze(result['test']['target']),
            'pred': y_pred,
            'subset': subset
        }

    def benchmark(self, dataset: HoldoutDataset):
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.evaluate)(dataset, 42 + i)
            for i in range(self.n_holdouts)
        )

        predictions = np.array([(i['pred'], i['target']) for i in results])
        subsets = np.array([i['subset'] for i in results])

        return {
            'predictions': predictions,
            'subsets': subsets
        }
