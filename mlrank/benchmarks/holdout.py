import numpy as np

from sklearn.utils.multiclass import type_of_target

from sklearn.externals.joblib import Parallel, delayed

from mlrank.preprocessing.dichtomizer import MaxentropyMedianDichtomizationTransformer, map_continuous_names, \
    DichtomizationIssue
from mlrank.submodular.optimization.optimizer import SubmodularOptimizer
from mlrank.utils import split_dataset
from .benchmark import Benchmark


class HoldoutBenchmark(Benchmark):
    def __init__(
            self,
            optimizer: SubmodularOptimizer,
            feature_selection_share: float,
            decision_function,
            n_holdouts: int,
            n_jobs: int
    ):
        if feature_selection_share >= 1 or feature_selection_share <= 0:
            raise Exception('feature_selection_share should be in range (0, 1).')

        super().__init__(optimizer, decision_function)

        self.feature_selection_share=feature_selection_share
        self.n_holdouts = n_holdouts
        self.n_jobs = n_jobs

    def evaluate(self, X_plain, X_transformed, y, seed):
        result = split_dataset(X_plain, X_transformed, y, seed, int(1))
        subset = self.optimizer.select(result['train']['plain'], result['train']['transformed'], result['train']['target'])

        y_pred = self.train_and_fit(subset, result['train']['transformed'], result['train']['target'], result['test']['target'])

        return {
            'target': np.squeeze(result['test']['target']),
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
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.evaluate)(X_train_plain, X_train_transformed, y_train, 42 + i)
            for i in range(self.n_holdouts)
        )

        predictions = np.array([(i['pred'], i['target']) for i in results])
        subsets = np.array([i['subset'] for i in results])

        return {
            'predictions': predictions,
            'subsets': subsets
        }


class DichtomizedHoldoutBenchmark(HoldoutBenchmark):
    def __init__(
            self,
            optimizer: SubmodularOptimizer,
            feature_selection_share: float,
            decision_function,
            n_holdouts: int,
            n_jobs: int,
            n_bins: int
    ):
        if not hasattr(decision_function, 'predict_proba'):
            raise Exception('decision function should have predict_proba attribute')

        super().__init__(optimizer, feature_selection_share, decision_function, n_holdouts, n_jobs)

        self.n_bins = n_bins

    def evaluate(self, X_plain, X_transformed, y, seed):
        try:
            result = split_dataset(X_plain, X_transformed, y, seed, int(1))

            if type_of_target(y) == 'continuous':
                dichtomizer = MaxentropyMedianDichtomizationTransformer(n_splits=self.n_bins)
                dichtomizer.fit(result['train']['target'].reshape(-1, 1))

                y_train = dichtomizer.transform(result['train']['target'].reshape(-1, 1))
                y_test = dichtomizer.transform(result['test']['target'].reshape(-1, 1))
            else:
                y_train = result['train']['target']
                y_test = result['test']['target']

            subset = self.optimizer.select(result['train']['plain'], result['train']['transformed'], y_train)
            y_pred = self.train_and_fit(subset, result['train']['transformed'], y_train, result['test']['transformed'])

            return {
                'target': np.squeeze(y_test),
                'pred': y_pred,
                'subset': subset
            }
        except DichtomizationIssue as e:
            print(e)
            return dict()
