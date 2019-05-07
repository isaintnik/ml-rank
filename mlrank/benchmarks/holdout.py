import numpy as np

from sklearn import clone
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target

from sklearn.externals.joblib import Parallel, delayed

from mlrank.preprocessing.dichtomizer import MaxentropyMedianDichtomizationTransformer, map_continious_names
from mlrank.submodularity.optimization.optimizer import SubmodularOptimizer


class HoldoutBenchmark(object):
    def __init__(self, optimizer: SubmodularOptimizer, feature_selection_share: float, decision_function, n_holdouts: int, n_jobs: int):
        if feature_selection_share >= 1 or feature_selection_share <= 0:
            raise Exception('feature_selection_share should be in range (0, 1).')

        self.optimizer=optimizer
        self.feature_selection_share=feature_selection_share
        self.decision_function=decision_function
        self.n_holdouts = n_holdouts
        self.n_jobs = n_jobs

    def predict_for_data(self, subset, X_complete, y_complete, X_train, X_test, y_train, y_test):
        model = clone(self.decision_function)
        model.fit(X_train[:, subset], np.squeeze(y_train))

        return model.predict(X_test[:, subset]), y_test

    def split_dataset(self, X, y, seed):
        X_complete, X_test, y_complete, y_test = train_test_split(X, y, test_size=int(1), random_state=seed)
        X_f, X_h, y_f, y_h = train_test_split(X_complete, y_complete, test_size=1 - self.feature_selection_share, random_state=seed)

        return X_complete, y_complete, X_h, y_h, X_f, y_f, X_test, y_test

    def evaluate(self, X, y, seed):
        X_complete, y_complete, X_h, y_h, X_f, y_f, X_test, y_test = self.split_dataset(X, y, seed)
        subset = self.optimizer.select(X_f, y_f)

        y_pred, y_test = self.predict_for_data(subset, X_complete, y_complete, X_h, y_h, X_test, y_test)
        return {
            'target': np.squeeze(y_test),
            'pred': y_pred,
            'subset': subset
        }

    def benchmark(self, X, y):
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.evaluate)(X, y, 42 + i)
            for i in range(self.n_holdouts)
        )

        predictions = np.array([(i['pred'], i['target']) for i in results])
        subsets = np.array([i['subset'] for i in results])

        return {
            'predictions': predictions,
            'subsets': subsets
        }


class DichtomizedHoldoutBenchmark(HoldoutBenchmark):
    def __init__(self, optimizer: SubmodularOptimizer, feature_selection_share: float, decision_function, n_holdouts: int, n_jobs: int, n_bins: int):
        if not hasattr(decision_function, 'predict_proba'):
            raise Exception('decision function should have predict_proba attribute')

        super().__init__(optimizer, feature_selection_share, decision_function, n_holdouts, n_jobs)

        self.n_bins = n_bins

    def predict_for_data(self, subset, X_complete, y_complete, X_train, y_train, X_test, y_test):
        # TODO: include features from feature selection in testing procedure or not?

        model = clone(self.decision_function)

        if type_of_target(y_complete) == 'continuous':
            dichtomizer = MaxentropyMedianDichtomizationTransformer(n_splits = self.n_bins)
            dichtomizer.fit(y_complete.reshape(-1, 1))

            target = np.squeeze(dichtomizer.transform(y_train.reshape(-1, 1)))
            model.fit(X_train[:, subset], np.squeeze(target))

            pred = np.squeeze(model.predict(X_test[:, subset]))
        else:
            model.fit(X_train[:, subset], np.squeeze(y_train))

            pred = np.squeeze(model.predict(X_test[:, subset]))


        return pred, y_test
