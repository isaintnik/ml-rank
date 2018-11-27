import numpy as np
from copy import deepcopy as copy

from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

from .dichtomization import MaxentropyMedianDichtomizationTransformer

class SomeFabulousTransformation(BaseEstimator):
    def __init__(self, base_estimator, dichtomized=False, n_splits=32, exhausitve=True, random_seed=42, verbose=1):
        self.base_estimator = base_estimator
        self.n_splits = n_splits
        self.random_seed = random_seed
        self.exhausitve = exhausitve
        self.verbose = verbose
        self.dichtomized = dichtomized
        self._feature_space = None
        self._feature_dichtomizers = None

    @staticmethod
    def _cross_entropy(p, q):
        p = np.array(p)
        q = np.array(q)
        q[q == 0] = 1e-8
        return -np.sum(p * np.log(q))

    @staticmethod
    def _synchronize_two_dicts(_from: dict, _to: dict):
        for i in np.setdiff1d(list(_from.keys()), list(_to.keys())):
            _to[i] = 0

    def _dichtomize(self, X):
        self._feature_dichtomizers = list()
        self._feature_space = list()

        if not self.dichtomized:
            for i in range(X.shape[1]):
                feature = X[:, i].reshape(-1, 1)
                dichtomizer = MaxentropyMedianDichtomizationTransformer(32).fit(feature)
                feature_dichtomized = dichtomizer.transform(feature)
                onehot_encoder = OneHotEncoder(sparse=True).fit(feature_dichtomized)

                self._feature_dichtomizers.append({'dichtomizer': dichtomizer, 'encoder': onehot_encoder})
                self._feature_space.append(
                    {'categorical': feature_dichtomized, 'binary': onehot_encoder.transform(feature_dichtomized)})
        else:
            for i in range(X.shape[1]):
                feature = X[:, i].reshape(-1, 1)
                onehot_encoder = OneHotEncoder(sparse=True).fit(feature)
                self._feature_dichtomizers.append({'dichtomizer': None, 'encoder': onehot_encoder})
                self._feature_space.append({'categorical': feature, 'binary': onehot_encoder.transform(feature)})

    def _calc_cross_entropy(self, ix_feature, pred, dataset_size):
        real = np.asarray(self._feature_space[ix_feature]['binary'].sum(1))
        pred = np.asarray(pred.sum(1))

        # TODO: уточнить
        pred[np.argwhere(pred == 2)] = 0

        pred_category, pred_counts = np.unique(pred, return_counts=True)
        real_category, real_counts = np.unique(real, return_counts=True)

        pred_proba = pred_counts / dataset_size
        real_proba = real_counts / dataset_size

        real_stats = dict(zip(real_category, real_proba))
        pred_stats = dict(zip(pred_category, pred_proba))

        SomeFabulousTransformation._synchronize_two_dicts(real_stats, pred_stats)

        return SomeFabulousTransformation._cross_entropy(list(real_stats.values()), list(pred_stats.values()))

    def _fit_transform(self, X, initial_feature_ix):
        dataset_size = X.shape[0]
        free_features_ix = [i for i in range(len(self._feature_space)) if i != initial_feature_ix]
        active_features_subset = [self._feature_space[initial_feature_ix]['binary']]
        active_features_subset_ix = [initial_feature_ix]

        while len(active_features_subset) != len(self._feature_space):
            max_entropy = -1
            max_entropy_feature_ix = -1
            max_entropy_feature_value = None

            if self.verbose > 1:
                print('currently processed {} features out of {}'.format(len(active_features_subset),
                                                                         len(self._feature_space)))
                print('number of active features {}'.format(len(active_features_subset)))

            for ix_current_feature in free_features_ix:
                if len(active_features_subset) > 1:
                    model_input_features = sparse.hstack(active_features_subset)
                else:
                    model_input_features = active_features_subset[0]

                estimator = copy(self.base_estimator)
                estimator.fit(model_input_features, self._feature_space[ix_current_feature]['categorical'].squeeze())

                pred = estimator.predict(model_input_features)

                pred_onehot = self._feature_dichtomizers[ix_current_feature]['encoder'].transform(pred.reshape(-1, 1))
                pred_diff = (pred_onehot != self._feature_space[ix_current_feature]['binary']).astype(np.int32)

                entropy = self._calc_cross_entropy(ix_current_feature, pred_diff, dataset_size)

                if entropy > max_entropy:
                    max_entropy_feature_value = pred_diff
                    max_entropy_feature_ix = ix_current_feature
                    max_entropy = entropy

            free_features_ix.remove(max_entropy_feature_ix)
            active_features_subset.append(max_entropy_feature_value)
            active_features_subset_ix.append(max_entropy_feature_ix)

        return np.hstack(active_features_subset), active_features_subset_ix

    def fit_transform(self, X):
        np.random.seed(self.random_seed)

        self._dichtomize(X)

        if not self.exhausitve:
            initial_feature_ix = np.random.randint(0, len(self._feature_space))
            return self._fit_transform(X, initial_feature_ix)

        features_subset = list()
        features_subset_ix = list()
        for initial_feature_ix in range(len(self._feature_space)):
            if self.verbose == 1:
                print('processing starting feature {}'.format(initial_feature_ix))
            features_subset, features_subset_ix = self._fit_transform(X, initial_feature_ix)

        return features_subset, np.hstack(features_subset_ix)
