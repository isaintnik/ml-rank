import numpy as np
from warnings import warn

from sklearn.metrics import mutual_info_score, log_loss
from sklearn.utils.multiclass import type_of_target
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.decomposition import FastICA
from scipy.stats import entropy

from mlrank.preprocessing.dichtomizer import map_continious_names, MaxentropyMedianDichtomizationTransformer

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


def informational_regularization_1(A, X_d, X_c, decision_function, n_bins=4):
    """
    Returns -R(X_A , X)
    A -> X --> -R(X_A, X) -> 0
    R(X_A, X) := \sum_{f \in F} D_{KL}(h_a, f)
    :param subset:
    :param X_d:
    :param X_c:
    :param decision_function:
    :param n_bins:
    :return:
    """
    X_subset = X_d[:, A]

    infosum = 0

    for i in range(X_c.shape[1]):
        model = clone(decision_function)

        r = X_c[:, i]

        model.fit(X_subset, r)

        dichtomizer = MaxentropyMedianDichtomizationTransformer(n_bins)
        dichtomizer.fit(r.reshape(-1, 1))

        r_d = dichtomizer.transform_ordered(r.reshape(-1, 1))
        p_d = dichtomizer.transform_ordered(model.predict(X_subset).reshape(-1, 1))

        print(r.min(), r.max())
        print(np.unique(model.predict(X_subset)))

        # предсказание часто уходит в граничные зоны

        a, p_proba = np.unique(p_d, return_counts=True)
        b, r_proba = np.unique(r_d, return_counts=True)

        if a.shape[0] != b.shape[0]:
            unque_elements = np.array([[b[i], r_proba[i]] for i in range(len(b)) if b[i] not in a])

            a = np.append(a, np.squeeze(unque_elements[:, 0]))
            p_proba = np.append(p_proba, np.array([0] * (r_proba.shape[0] - p_proba.shape[0])))

            a = a.tolist()
            b = b.tolist()

            p_proba = p_proba.tolist()
            r_proba = r_proba.tolist()

            a, p_proba = list(zip(*sorted(zip(a, p_proba), key=lambda x: x[0])))
            b, r_proba = list(zip(*sorted(zip(b, r_proba), key=lambda x: x[0])))

        #print(p_proba, r_proba)
        infosum += entropy(p_proba, r_proba)

    return -infosum / (entropy((1, 0, 0, 0), (1/4, 1/4, 1/4, 1/4)) * X_d.shape[0])


def informational_regularization_2(A, X_d, X_c, decision_function, n_bins=4):
    """
    Returns -R(X_A , X)
    A -> X --> -R(X_A, X) -> 0
    R(X_A, X) := \sum_{f \in F} D_{KL}(h_a, f)
    :param subset:
    :param X_d:
    :param X_c:
    :param decision_function:
    :param n_bins:
    :return:
    """
    if not A:
        return 0

    X_subset = X_d[:, A]

    infosum = list()
    nonneg_offset = list()

    for i in range(X_c.shape[1]):
        model = clone(decision_function)

        r = X_c[:, i]

        model.fit(X_subset, r)

        # TODO: could be precalculated
        dichtomizer = MaxentropyMedianDichtomizationTransformer(n_bins)
        dichtomizer.fit(r.reshape(-1, 1))

        # TODO: could be precalculated
        r_d = np.squeeze(dichtomizer.transform_ordered(r.reshape(-1, 1)))
        p_d = np.squeeze(dichtomizer.transform_ordered(model.predict(X_subset).reshape(-1, 1)))

        continious_labels = np.unique(r_d).tolist()

        binarizer = LabelBinarizer()
        binarizer.fit(np.unique(map_continious_names(r_d, continious_labels)))

        a = binarizer.transform(map_continious_names(r_d, continious_labels))
        b = binarizer.transform(map_continious_names(p_d, continious_labels))

        c = binarizer.transform([1]*X_c.shape[0])

        _, counts = np.unique(r_d, return_counts=True)

        #infosum.append(log_loss(a, c) - log_loss(a, b))
        infosum.append(log_loss(a, b))
        nonneg_offset.append(log_loss(a, c))

    return np.mean(infosum) / X_c.shape[1]# - np.mean(nonneg_offset)


