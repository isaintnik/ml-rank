import numpy as np
import warnings

from sklearn.metrics import mutual_info_score, log_loss
from sklearn.utils.multiclass import type_of_target
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, LabelEncoder
from sklearn.decomposition import FastICA
from scipy.stats import entropy

from mlrank.preprocessing.dichtomizer import map_continuous_names, MaxentropyMedianDichtomizationTransformer, dichtomize_matrix

from pyitlib.discrete_random_variable import entropy_joint

from itertools import product


# TODO: not sure how this algorithm works, find it out or rewrite it
from mlrank.utils import make_features_matrix, get_model_classification_order, fix_target


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


#def informational_regularization_classification(A, X_f, X_t, decision_function) -> float:
#    """
#    Returns R(X_A , X)
#    A -> X --> R(X_A, X) -> 0
#    R(X_A, X) := \sum_{f \in F} I(h_A, f)
#    :param A: indices of subset features
#    :param X_f: raw features
#    :param X_t: dichtomized features for classification task
#    :param decision_function: F:X -> C, C = vector of integers
#    :return: float
#    """
#
#    # I(\emptyset, x) = 0
#    if not A:
#        return 0
#
#    infosum = list()
#
#    for i in range(X_f.shape[1]):
#        model = clone(decision_function)
#
#        r = X_t[:, i]
#
#        if np.unique(r).shape[0] > 1:
#            model.fit(X_f[:, A], r)
#
#            r_d = np.squeeze(r)
#            p_d = np.squeeze(model.predict(X_f[:, A]))
#        else:
#            # constant model
#            r_d = p_d = np.squeeze(r)
#
#        infosum.append(mutual_info_score(r_d, p_d))
#
#    return np.sum(infosum)


#def log_likelihood_cross_features(A, X_f, X_t, decision_function, n_random_iter=20, eps_norm = 1e-8) -> float:
#    f_lls = list()
#
#    for i in range(X_f.shape[1]):
#        model = clone(decision_function)
#
#        lencoder = LabelEncoder()
#        decision_function = clone(decision_function)
#
#        y = X_t[:, i]
#        y_arange = np.arange(y.size)
#        y_labels = lencoder.fit_transform(y)
#
#        if A:
#            if np.unique(y).shape[0] > 1:
#                model.fit(X_f[:, A], y)
#
#                y_pred = model.predict_proba(X_f[:, A])
#                ll = np.sum(np.log(y_pred[y_arange, np.squeeze(y_labels)] + eps_norm))
#            else:
#                # in case of constant model log likelihood = log(0) = 1
#                ll = 0
#        else:
#            lls = list()
#            for i in range(n_random_iter):
#                y_pred = np.random.beta(1 / 2, 1 / 2, size=len(y))
#                lls.append(np.sum(np.log(y_pred + eps_norm)))
#            ll = np.mean(lls)
#
#        f_lls.append(ll)
#
#    return np.sum(f_lls)


def log_likelihood_cross_features(
        A: list, X_f: dict, X_f_test: dict,
        X_t: dict, X_t_test: dict,
        decision_function,
        n_random_iter=20, eps_norm = 1e-8
) -> float:
    f_lls = list()

    for i in X_t.keys():
        #print('->', i)
        model = clone(decision_function)
        decision_function = clone(decision_function)

        #X_train_m = make_features_matrix(X_train, A)

        y = X_t[i]
        y_test = np.copy(X_t_test[i])  # not optimal but dunno how to do a better way

        if A:
            if np.unique(y).shape[0] > 1:
                X_train = make_features_matrix(X_f, A)
                X_test = make_features_matrix(X_f_test, A)

                model.fit(X_train, y)
                y_pred = model.predict_proba(X_test)

                # map test values to indices to calc log likelihood
                classes_ = get_model_classification_order(model)
                y_test, y_pred = fix_target(classes_, y_test, y_pred)

                # log likelihood
                ll = np.sum(np.log(y_pred[np.arange(y_test.size), np.squeeze(y_test)] + eps_norm)) # потенциальная ошибка
            else:
                # in case of constant model log likelihood = log(0) = 1
                ll = 0
        else:
            lls = list()
            for i in range(n_random_iter):
                y_pred = np.random.beta(1 / 2, 1 / 2, size=len(y))
                lls.append(np.sum(np.log(y_pred + eps_norm)))
            ll = np.mean(lls)

        f_lls.append(ll)

    return float(np.sum(f_lls))
