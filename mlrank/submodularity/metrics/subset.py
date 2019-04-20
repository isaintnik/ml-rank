import numpy as np
import warnings

from sklearn.metrics import mutual_info_score, log_loss
from sklearn.utils.multiclass import type_of_target
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.decomposition import FastICA
from scipy.stats import entropy

from mlrank.preprocessing.dichtomizer import map_continious_names, MaxentropyMedianDichtomizationTransformer, dichtomize_matrix

from pyitlib.discrete_random_variable import entropy_joint

from itertools import product
from functools import reduce



# assumes that features are independently distributed
# could be precomputed O(n^2) or realtime O(k*n)
def mean_pairwise_score(subset, X, decision_function):
    feature_scores = 0
    for i in subset:
        score = 0
        for j in range(X.shape[1]):
            if j == i:
                continue
            df = clone(decision_function)

            score += mutual_info_score(df.fit_predict(X[:, j]), X[:, i])
        feature_scores += score / (X.shape[1] - 1)
    return feature_scores


# should be increasing at each iterathing + unknown whether is this submodular or not
# O(m*(n-m)!) ?
def greedy_subset_score(subset, X, decision_function):
    new_subset = list()
    subset_score = 0

    while len(new_subset) != len(subset):
        max_subset_feature_score = -1
        max_subset_feature_index = -1

        for feature in subset:
            if feature in new_subset:
                continue

            max_score = -1
            max_index = -1

            for i in range(X.shape[1]):
                if i == feature or i in new_subset:
                    continue

                df = clone(decision_function)

                encoder = OneHotEncoder(sparse=False)
                x = encoder.fit_transform(X[:, new_subset + [feature]])
                y = encoder.fit_transform(X[:, i])

                score = mutual_info_score(df.fit_predict(x, y))

                if score > max_score:
                    max_score = score
                    max_index = i

            if max_subset_feature_score > max_score:
                max_subset_feature_score = max_score
                max_subset_feature_index = max_index

        new_subset.append(max_subset_feature_index)
        subset_score += max_subset_feature_score

    return subset_score


# H(X \ {subset} | {subset}) = H(X) - H({subset})
# I(X, {subset}) = H(X) - H(X | {subset}) = H({subset}) - submodular (conditionally, except xor case)

# TODO: not sure how this algorithm works, find it out or rewrite it
def joint_entropy_score_estimate(subset, X):
    features = list()
    for i in subset:
        features.append(map_continious_names(np.squeeze(X[:, i])))
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


def informational_regularization_regression(A, X, decision_function, n_bins=4):
    """
    Returns R(X_A , X)
    A -> X --> R(X_A, X) -> 0
    R(X_A, X) := \sum_{f \in F} H(h_A, f)
    :param A: indices of subset features
    :param X: continious data
    :param decision_function:
    :param n_bins:
    :return:
    """
    if not A:
        return 0

    infosum = list()

    n_features = 0
    for i in range(X.shape[1]):
        model = clone(decision_function)

        r = X[:, i]

        if type_of_target(r) != 'continuous':
            warnings.warn(f"Binary features are not supported in continuous target. \n {i}-th feature in dataset is ignored")
            continue

        model.fit(X[:, A], r)

        dichtomizer = MaxentropyMedianDichtomizationTransformer(n_bins)
        dichtomizer.fit(r.reshape(-1, 1))

        r_d = np.squeeze(dichtomizer.transform_ordered(r.reshape(-1, 1)))
        p_d = np.squeeze(dichtomizer.transform_ordered(model.predict(X[:, A]).reshape(-1, 1)))

        continious_labels = np.unique(r_d).tolist()

        binarizer = LabelBinarizer()
        binarizer.fit(np.unique(map_continious_names(r_d, continious_labels)))

        a = binarizer.transform(map_continious_names(r_d, continious_labels))
        b = binarizer.transform(map_continious_names(p_d, continious_labels))

        infosum.append(log_loss(a, b))
        #infosum.append(mutual_info_score(
        #    map_continious_names(r_d, continious_labels),
        #    map_continious_names(p_d, continious_labels)
        #))

        n_features += 1

    return np.mean(infosum) / n_features


def informational_regularization_classification(A, X, decision_function, n_bins=4):
    """
    Returns R(X_A , X)
    A -> X --> R(X_A, X) -> 0
    R(X_A, X) := \sum_{f \in F} H(h_A, f)
    :param A: indices of subset features
    :param X: continious data
    :param decision_function:
    :param n_bins:
    :return:
    """
    if not A:
        return 0

    infosum = list()

    for i in range(X.shape[1]):
        model = clone(decision_function)

        r = X[:, i]

        if type_of_target(r) == 'continuous':
            dichtomizer = MaxentropyMedianDichtomizationTransformer(n_bins)
            dichtomizer.fit(r.reshape(-1, 1))

            r_d = np.squeeze(dichtomizer.transform_ordered(r.reshape(-1, 1)))
            r_d = map_continious_names(r_d)

            model.fit(X[:, A], r_d)

            p_d = model.predict(X[:, A])
        else:
            model.fit(X[:, A], r)

            r_d = np.squeeze(r)
            p_d = np.squeeze(model.predict(X[:, A]))

        binarizer = LabelBinarizer()
        binarizer.fit(np.unique(r_d))

        a = binarizer.transform(r_d)
        b = binarizer.transform(p_d)

        infosum.append(log_loss(a, b))

    return np.mean(infosum) / X.shape[1]
