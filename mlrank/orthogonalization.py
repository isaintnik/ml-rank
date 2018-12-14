import numpy as np
from copy import deepcopy as copy

from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.multiclass import type_of_target
from sklearn.utils import deprecated
from scipy import sparse

from mlrank.dichtomization import MaxentropyMedianDichtomizationTransformer


def cross_entropy_discrete(p, q):
    """
    Simple pity cross entropy implementation
    :param p: base distribution
    :param q: target distribution
    :return:
    """
    p = np.array(p)
    q = np.array(q)
    return -np.sum(p * np.log(q))

@deprecated(extra='entropy calculation approach is changed')
def synchronize_two_dicts(_a: dict, _b: dict):
    """
    helper method to synchronize keys in two dictionaries
    :param _a: source dictionary
    :param _b:  target dictionary
    :return:
    """
    for i in np.setdiff1d(list(_a.keys()), list(_b.keys())):
        if i not in _a.keys():
            _a[i] = 0

        if i not in _b.keys():
            _b[i] = 0

@deprecated(extra='invalid entropy function')
def calc_cross_entropy_from_binary_features(real, pred):
    """
    Calculate cross entropy between feature and it's prediction
    :param ix_feature: feature index in dataset
    :param pred: predicted value
    :param dataset_size:
    :return:
    """
    real = np.asarray(real.sum(1)) # [1, 1, 1, ..., 1]
    pred = np.asarray(pred.sum(1)) # [1, 0, 0, ..., 1]

    pred[np.argwhere(pred == 2)] = 1

    pred_category, pred_counts = np.unique(pred, return_counts=True)
    real_category, real_counts = np.unique(real, return_counts=True)

    pred_proba = pred_counts / real.shape[0]
    real_proba = real_counts / real.shape[0]

    real_stats = dict(zip(real_category, real_proba))
    pred_stats = dict(zip(pred_category, pred_proba))

    synchronize_two_dicts(real_stats, pred_stats)

    return cross_entropy_discrete(list(real_stats.values()), list(pred_stats.values()))


def cross_entropy_from_probas(real, pred):
    if hasattr(real, 'todense'):
        # for sparse matrices (sparse encoding of real)
        result = real.multiply(np.log(pred))
        # by some reason it returns COO
        result = result.tocsr()
    else:
        # for arrays (dense encoding)
        result = real * np.log(pred)

    result = np.nan_to_num(result)
    result[result == -np.inf] = 0
    return -result.sum()


class DichtomizedTransformer(object):
    def __init__(self, dichtomized=False, n_splits=32):
        """
        Dichtomization of features and target
        :param dichtomized: whether dataset is dichtomized or not
        :param n_splits: number of splits to dichtomize continuous variables
        """
        self.n_splits = n_splits
        self.dichtomized = dichtomized

        self._feature_space = None
        self._feature_dichtomizers = None

        self._target_categorical = None
        self._target_onehot = None
        self._target_encoder = None

    def _dichtomize_features(self, X):
        """
        Dichtomize all the features in the dataset
        assuming all of them are continuous
        :param X: source dataset (numpy matrix where columns are features)
        :return:
        """
        self._feature_dichtomizers = list()
        self._feature_space = list()

        if not self.dichtomized:
            for i in range(X.shape[1]):
                # TODO: check whether the feature is continious
                feature = X[:, i].reshape(-1, 1)
                dichtomizer = MaxentropyMedianDichtomizationTransformer(self.n_splits).fit(feature)
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

    def _dichtomize_target(self, y):
        """
        Dichtomize dataset target if it is continuous
        assuming all of them are continuous
        :param X: source dataset (numpy matrix where columns are features)
        :return:
        """
        target_type = type_of_target(y)

        self._target_encoder = OneHotEncoder(sparse=True)

        if target_type == 'continuous':
            self._target_categorical = MaxentropyMedianDichtomizationTransformer(self.n_splits).fit_transform(y)
        elif target_type in ['binary', 'multiclass']:
            self._target_categorical = y
        else:
            raise Exception('Target type is incompatible with the model')

        self._target_onehot = sparse.csr_matrix(self._target_encoder.fit_transform(self._target_categorical))


class MLRankTransformer(BaseEstimator, DichtomizedTransformer):
    def __init__(self,
                 base_estimator,
                 dichtomized=False,
                 n_splits=32,
                 exhausitve=True,
                 random_seed=42,
                 verbose=1,
                 decision_boundary=.5):
        """
        First implementation of MLRank
        - this version searches ranks the input variables according to the information that they share in the dataset
        :param base_estimator:
        :param dichtomized: whether dataset is dichtomized or not
        :param n_splits: number of splits to dichtomize continuous variables
        :param exhausitve: exhaustively search for optimal subset through all the variables
        :param random_seed: seed for random initialization of initial feature
        :param verbose: whether output debug information or not
        """
        super().__init__(dichtomized, n_splits)

        self.base_estimator = base_estimator
        self.random_seed = random_seed
        self.exhausitve = exhausitve
        self.decision_boundary = decision_boundary
        self.verbose = verbose

    def _fit_transform(self, initial_feature_ix):
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

                estimator.fit(model_input_features,
                              self._feature_space[ix_current_feature]['categorical'].squeeze())

                pred_proba = estimator.predict(model_input_features)
                real = self._feature_space[ix_current_feature]['binary']
                entropy = cross_entropy_from_probas(real, pred_proba)

                pred_onehot = sparse.csr_matrix(pred_proba > self.decision_boundary).astype(np.int32)
                pred_diff = (pred_onehot != real).astype(np.int32)

                if entropy > max_entropy:
                    max_entropy_feature_value = pred_diff
                    max_entropy_feature_ix = ix_current_feature
                    max_entropy = entropy

            free_features_ix.remove(max_entropy_feature_ix)
            active_features_subset.append(max_entropy_feature_value)
            active_features_subset_ix.append(max_entropy_feature_ix)

        # remove first value to remove multicollinearity
        # TODO: this method will kill the data if it is already properly dichtomized!!!
        active_features_subset = [x_i[:, 1:] for x_i in active_features_subset]

        return sparse.hstack(active_features_subset), active_features_subset_ix

    def fit_transform(self, X, y=None):
        np.random.seed(self.random_seed)

        self._dichtomize_features(X)
        if not self.exhausitve:
            initial_feature_ix = np.random.randint(0, len(self._feature_space))
            return self._fit_transform(initial_feature_ix)

        features_subset = list()
        features_subset_ix = list()
        for initial_feature_ix in range(len(self._feature_space)):
            if self.verbose == 1:
                print('processing starting feature {}'.format(initial_feature_ix))
            _features_subset, _features_subset_ix = self._fit_transform(initial_feature_ix)

            features_subset.append(_features_subset)
            features_subset_ix.append(_features_subset_ix)

        return features_subset, np.vstack(features_subset_ix)


class MLRankTargetBasedTransformer(BaseEstimator, DichtomizedTransformer):
    def __init__(self,
                 base_estimator,
                 transformation,
                 dichtomized=False,
                 n_splits=32,
                 random_seed=42,
                 verbose=1,
                 decision_boundary=.5,
                 use_max_entropy=True):
        """
        :param base_estimator:
        :param dichtomized: whether dataset is dichtomized or not
        :param n_splits: number of splits to dichtomize continuous variables
        :param random_seed: seed for random initialization of initial feature
        :param verbose: whether output debug information or not
        """
        super().__init__(dichtomized, n_splits)

        if not hasattr(base_estimator, 'predict_proba'):
            raise Exception('ml-rank requires probabilistic model')

        self.decision_boundary = decision_boundary
        self.use_max_entropy = use_max_entropy
        self.base_estimator = base_estimator
        self.random_seed = random_seed
        self.verbose = verbose
        self.transformation = transformation

    def _fit_transform(self):
        from sklearn.metrics import accuracy_score
        free_features_ix = [i for i in range(len(self._feature_space))]
        active_features_subset = []
        active_features_subset_ix = []
        entropy_prev = None

        while len(active_features_subset) != len(self._feature_space):
            entropy_current = None
            if self.use_max_entropy:
                entropy_current = -np.inf
            else:
                entropy_current = np.inf

            entropy_feature_ix = -1
            entropy_feature_value = None

            for ix_current_feature in free_features_ix:
                if len(active_features_subset) > 1:
                    model_input_features = sparse.hstack(active_features_subset)
                else:
                    model_input_features = self._feature_space[ix_current_feature]['binary']

                estimator = copy(self.base_estimator)

                estimator.fit(model_input_features, self._target_categorical)

                pred_proba = estimator.predict_proba(model_input_features)
                entropy = cross_entropy_from_probas(self._target_onehot.astype(np.float32), pred_proba)

                pred_onehot = sparse.csr_matrix(pred_proba > self.decision_boundary).astype(np.int32)
                pred_diff = self.transformation(pred_onehot, self._target_onehot).astype(np.int32)


                if self.use_max_entropy and (entropy > entropy_current) or not self.use_max_entropy and (entropy < entropy_current):
                    entropy_feature_value = pred_diff
                    entropy_feature_ix = ix_current_feature
                    entropy_current = entropy

            # if there is no changes in entropy, return given dataset
            # since there is no point to continue
            if entropy_prev is None or entropy_prev != entropy_current:
                entropy_prev = entropy_current
                free_features_ix.remove(entropy_feature_ix)
                active_features_subset.append(entropy_feature_value)
                active_features_subset_ix.append(entropy_feature_ix)
            else:
                break

        return sparse.hstack(active_features_subset), active_features_subset_ix

    def fit_transform(self, X, y=None):
        np.random.seed(self.random_seed)

        self._dichtomize_features(X)
        self._dichtomize_target(y)

        return self._fit_transform()