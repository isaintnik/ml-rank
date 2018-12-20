import numpy as np

from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats

from tqdm import tqdm

from .optimal_subset_alrorithm import *


class OptimalSubsetBenchmark(object):
    def __init__(self,
                 optimal_subset_algorithm: OSAlgorithm,
                 n_holdout_validations: int,
                 metric=accuracy_score):
        self.n_holdout_validations = n_holdout_validations
        self.os_algorithm = optimal_subset_algorithm
        self.metric = metric

        self.stats = None

    def get_stats(self):
        return {
            "nobs": len(self.stats),
            "raw": self.stats,
            "mean": np.mean(self.stats),
            "median": np.median(self.stats),
            "variance": np.var(self.stats),
            **{f"percentile_{k}": np.percentile(self.stats, k) for k in range(5, 95, 5)},
        }

    def benchmark(self, X, y):
        self.stats = list()
        for i in tqdm(range(self.n_holdout_validations)):
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, shuffle=True)

            self.os_algorithm.fit(X_train, y_train)
            ix_features = self.os_algorithm.get_important_features()
            # grab estimator from os algorithm
            estimator = self.os_algorithm.estimator
            estimator.fit(X_train[:, ix_features], y_train)

            score = self.metric(estimator.predict(X_val[:, ix_features]), y_val)
            self.stats.append(score)

        return self#stats.describe(self.stats)
