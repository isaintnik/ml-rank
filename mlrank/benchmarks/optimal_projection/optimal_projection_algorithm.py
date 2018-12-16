from abc import ABCMeta, abstractmethod

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding


class OPAlgorithm(object):
    __metaclass__ = ABCMeta

    def __init__(self, n_projections: int = None):
        self.n_projections = n_projections

    def set_projections_num(self, n_projections):
        self.n_projections = n_projections

    @abstractmethod
    def fit_transform(self, X, y = None): pass


class PCAWrap(OPAlgorithm):
    def __init__(self, n_projections: int = None):
        super().__init__(n_projections)

    def fit_transform(self, X, y = None):
        return PCA(n_components=self.n_projections).fit_transform(X)


class ICAWrap(OPAlgorithm):
    def __init__(self, n_projections: int = None):
        super().__init__(n_projections)

    def fit_transform(self, X, y=None):
        return FastICA(n_components=self.n_projections).fit_transform(X)


class TSNEWrap(OPAlgorithm):
    def __init__(self, n_projections: int = None):
        if n_projections is not None and n_projections > 3:
            print('tsne is REALLY uneffective with n_projections > 3')

        super().__init__(n_projections)

    def set_projections_num(self, n_projections):
        if n_projections > 3:
            print('tsne is REALLY uneffective with n_projections > 3')

        self.n_projections = n_projections

    def fit_transform(self, X, y=None):
        return TSNE(n_components=self.n_projections).fit_transform(X)


class LLEWrap(OPAlgorithm):
    def __init__(self, n_projections: int = None):
        super().__init__(n_projections)

    def fit_transform(self, X, y=None):
        return LocallyLinearEmbedding(n_components=self.n_projections).fit_transform(X)
