from abc import ABCMeta, abstractmethod
from copy import deepcopy
import numpy as np

from lightgbm import LGBMClassifier

from mlrank.hyperparams_opt import (
    bayesian_optimization_lightgbm, get_optimized_lightgbm_gbdt, get_optimized_lightgbm_rf
)

from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector, ExhaustiveFeatureSelector

from mlrank.orthogonalization import MLRankTargetBasedTransformer

SCORING_METHOD = 'accuracy'


class OSAlgorithm(object):
    __metaclass__ = ABCMeta

    def __init__(self, model, n_features: int, params = dict()):
        self.estimator = model
        self.n_features = n_features

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def get_important_features(self):
        pass


class SFSWrap(OSAlgorithm):
    def __init__(self, model, n_features: int, params = dict()):
        super().__init__(deepcopy(model), n_features)
        self.n_features = n_features
        self.os_model = SequentialFeatureSelector(self.estimator, n_features, scoring=SCORING_METHOD)

    def fit(self, X, y):
        self.os_model.fit(X, y)

    def get_important_features(self):
        feature_names = self.os_model.subsets_[self.n_features]['feature_names']
        return [int(i) for i in feature_names]


class RFEWrap(OSAlgorithm):
    def __init__(self, model, n_features: int, params = dict()):
        super().__init__(deepcopy(model), n_features)
        self.n_features = n_features
        self.os_model = SequentialFeatureSelector(self.estimator, n_features, scoring=SCORING_METHOD, forward=False)

    def fit(self, X, y):
        self.os_model.fit(X, y)

    def get_important_features(self):
        feature_names = self.os_model.subsets_[self.n_features]['feature_names']
        return [int(i) for i in feature_names]


class EFSWrap(OSAlgorithm):
    def __init__(self, model, n_features: int, params = dict()):
        super().__init__(deepcopy(model), n_features)
        self.os_model = ExhaustiveFeatureSelector(self.estimator, n_features)

    def fit(self, X, y):
        self.os_model.fit(X, y)

    def get_important_features(self):
        feature_names = self.os_model.best_feature_names_
        return [int(i) for i in feature_names]


class LRCoefficentsWrap(OSAlgorithm):
    def __init__(self, model, n_features: int, params = dict()):
        super().__init__(deepcopy(model), n_features)

        self.os_model = LogisticRegression(C=1e10)

    def fit(self, X, y):
        self.os_model.fit(X, y)

    def get_important_features(self):
        # TODO: check validity of this method
        model_coefs = np.abs(self.os_model.coef_).sum(0)
        top_features = sorted(zip(model_coefs.tolist(), range(model_coefs.shape[0])), key=lambda x: -x[0])[:5]
        feature_names = [i[1] for i in top_features]
        return feature_names


class RFImportancesWrap(OSAlgorithm):
    def __init__(self, model, n_features: int, params = dict()):
        super().__init__(deepcopy(model), n_features)
        self.os_model = None

    def fit(self, X, y):
        self.os_model = get_optimized_lightgbm_rf(X, y)
        if type(self.estimator) == 'LGBMClassifier':
            self.estimator = self.os_model

        self.os_model.fit(X, y)

    def get_important_features(self):
        fi = self.os_model.feature_importances_
        return np.argsort(fi)[::-1]


class MLRankWrap(OSAlgorithm):
    def __init__(self, model, n_features: int, params = dict()):
        super().__init__(deepcopy(model), n_features)
        self.os_model = MLRankTargetBasedTransformer(base_estimator=self.estimator, **params)

    def fit(self, X, y):
        _, self.important_features = self.os_model.fit_transform(X, y)

    def get_important_features(self):
        return self.important_features[:self.n_features]
