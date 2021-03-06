import numpy as np

from joblib import Parallel, delayed

from sklearn.base import clone

from mlrank.preprocessing.dichotomizer import DichotomizationIssue
from mlrank.submodular.optimization.optimizer import SubmodularOptimizer
from mlrank.utils import split_dataset

#from guppy import hpy; h = hpy()
#import objgraph


class ForwardFeatureSelection(SubmodularOptimizer):
    def __init__(self,
                 decision_function,
                 train_share: float = 1.0,
                 n_bins: int = 4,
                 n_features: int = -1,
                 n_cv_ffs: int = 1,
                 n_jobs=-1
                 ):
        """
        Perform greedy algorithm of feature selection ~ O(n_features ** 2)
        :param decision_function: decision function to be evaluated
        :param score_function: score function for submodular optimization
        :param train_share: share of data to be trained on
        :param n_cv_ffs: number of CV's, 1 = evaluate on training set
        :param n_bins: only used for continuous targets
        """
        super().__init__()

        self.decision_function = decision_function
        self.n_cv_ffs = n_cv_ffs
        self.n_bins = n_bins
        self.train_share = train_share
        self.n_features = n_features
        self.logs = None
        self.n_jobs = n_jobs

        self.seeds = [(42 + i) for i in range(self.n_cv_ffs)]

    def evaluate_new_feature(self, prev_subset, new_feature, X_f, X_t, y):
        raise NotImplementedError()

    def get_dichotomized(self, X_plain: dict, X_transformed: dict, y: np.array, continuous_feature_list: list):
        try:
            X_f = X_transformed
            X_t = self.dichotomize_features(X_plain, self.n_bins, continuous_feature_list)
            y = self.dichotomize_target(y, self.n_bins)
        except Exception as e:
            print(e)
            raise DichotomizationIssue(self.n_bins)

        return X_f, X_t, y

    def select(self, X_plain: dict, X_transformed: dict, y: np.array, continuous_feature_list: list) -> list:
        X_f, X_t, y = self.get_dichotomized(X_plain, X_transformed, y, continuous_feature_list)

        subset = list()
        subset_logs = list()
        if self.n_features == -1:
            self.n_features = len(X_plain.keys())
        self.logs = list()

        prev_top_score = -np.inf

        feature_names = list(X_plain.keys())
        for _ in feature_names:
            feature_scores = list()

            for j in feature_names:
                if j in subset_logs:
                    feature_scores.append(-np.inf)
                else:
                    feature_scores.append(self.evaluate_new_feature(subset, j, X_f, X_t, y))

            top_feature = int(np.argmax(feature_scores))  # np.atleast_1d(np.squeeze(np.argmax(feature_scores)))[0]

            if np.max(feature_scores) > prev_top_score or self.n_features < len(X_plain.keys()):
                subset.append(feature_names[top_feature])
                prev_top_score = np.max(feature_scores)
            else:
                break

            subset_logs.append(feature_names[top_feature])

            self.logs.append({
                'subset': np.copy(subset_logs).tolist(),
                'score': np.max(feature_scores)
            })
        return subset

    def get_logs(self):
        return self.logs


class ForwardFeatureSelectionExtended(ForwardFeatureSelection):
    def __init__(self,
                 decision_function,
                 score_function,
                 train_share: float = 1.0,
                 n_bins: int = 4,
                 n_features: int = -1,
                 n_cv_ffs: int = 1,
                 n_jobs: int = -1,
                 ):
        """
        Perform greedy algorithm of feature selection ~ O(n_features ** 2)
        :param decision_function: decision function to be evaluated
        :param score_function: score function for submodular optimization
        :param train_share: share of data to be trained on
        :param n_cv_ffs: number of CV's, 1 = evaluate on training set
        :param n_bins: only used for continuous targets
        """
        super().__init__(
            decision_function,
            train_share,
            n_bins,
            n_features,
            n_cv_ffs,
            n_jobs
        )

        self.score_function = score_function

    def evaluate_new_feature(self, prev_subset: list, new_feature, X_f: dict, X_t: dict, y: np.array) -> float:
        A = prev_subset + [new_feature]
        scores = list()
        if self.n_jobs > 1:
            scores = Parallel(n_jobs=self.n_jobs)(
                delayed(self.score_function)(
                    A=A,
                    X_f=result['train']['transformed'],
                    X_f_test=result['test']['transformed'],
                    X_t=result['train']['plain'],
                    X_t_test=result['test']['plain'],
                    y=result['train']['target'],
                    y_test=result['test']['target'],
                    decision_function=clone(self.decision_function)
                )
                for result in (
                    split_dataset(X_t, X_f, y, self.seeds[i], 1 - self.train_share)
                    for i in range(self.n_cv_ffs)
                )
            )
        else:
            for i in range(self.n_cv_ffs):
                result = split_dataset(X_t, X_f, y, self.seeds[i], 1 - self.train_share)

                scores.append(self.score_function(
                    A=A,
                    X_f=result['train']['transformed'],
                    X_f_test=result['test']['transformed'],
                    X_t=result['train']['plain'],
                    X_t_test=result['test']['plain'],
                    y=result['train']['target'],
                    y_test=result['test']['target'],
                    decision_function=self.decision_function
                ))

        return float(np.mean(scores))
