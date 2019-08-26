import numpy as np
import warnings

from sklearn.metrics import mutual_info_score, log_loss
from sklearn.utils.multiclass import type_of_target
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.decomposition import FastICA
from scipy.stats import entropy

from mlrank.preprocessing.dichtomizer import map_continuous_names, MaxentropyMedianDichtomizationTransformer, dichtomize_matrix

from pyitlib.discrete_random_variable import entropy_joint

from itertools import product


# TODO: not sure how this algorithm works, find it out or rewrite it
def joint_entropy_score_estimate(subset, X):
    features = list()
    for i in subset:
        features.append(map_continuous_names(np.squeeze(X[:, i])))
    X_categorical = np.vstack(features)

    return entropy_joint(X_categorical)


# https://arxiv.org/pdf/1612.00554.pdf
def joint_entropy_score_ica_estimate(subset, X):
    n = X.shape[0]
    X_subset = X[:, subset]
    ica = FastICA()
    S = ica.fit_transform(X_subset)
    h = 0

    for s in S.T:
        s_tag, s_count = np.unique(s, return_counts=True)

        p = s_count / n
        h += np.sum(-p * np.log(p))

    return h - np.linalg.norm(ica.components_)


def joint_entropy_score_exact(subset, X):
    sub_X = X[:, subset]
    subset_len = len(subset)
    h = 0
    total_values = X.shape[0] * X.shape[1]

    for values in product(*[set(x) for x in sub_X.T]):
        p = np.sum(np.sum(sub_X[:, j] == values[j]) for j in range(subset_len)) / total_values
        h -= p * np.log2(p) if p != 0 else 0
    return h


def informational_regularization_classification(A, X_f, X_t, decision_function) -> float:
    """
    Returns R(X_A , X)
    A -> X --> R(X_A, X) -> 0
    R(X_A, X) := \sum_{f \in F} I(h_A, f)
    :param A: indices of subset features
    :param X_f: raw features
    :param X_t: dichtomized features for classification task
    :param decision_function: F:X -> C, C = vector of integers
    :return: float
    """

    # I(\emptyset, x) = 0
    if not A:
        return 0

    infosum = list()

    for i in range(X_f.shape[1]):
        model = clone(decision_function)

        r = X_t[:, i]

        if np.unique(r).shape[0] > 1:
            model.fit(X_f[:, A], r)

            r_d = np.squeeze(r)
            p_d = np.squeeze(model.predict(X_f[:, A]))
        else:
            # constant model
            r_d = p_d = np.squeeze(r)

        infosum.append(mutual_info_score(r_d, p_d))

    return np.mean(infosum) / X_f.shape[1]
