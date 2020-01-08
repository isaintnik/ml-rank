import numpy as np

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.utils.multiclass import type_of_target
from sklearn.base import clone

from mlrank.submodular.metrics.coefs import get_model_n_coefs
from mlrank.utils import make_features_matrix, get_model_classification_order, fix_target


#def mutual_information_classification(A, X, y, decision_function):
#    X = X[:, A]
#
#    target = np.squeeze(y)
#    target_type = type_of_target(target)
#
#    if target_type not in ['binary', 'multiclass']:
#        raise Exception(target_type, 'not supported.')
#
#    if np.unique(target).shape[0] > 1:
#        df = clone(decision_function)
#        df.fit(X, target)
#
#        pred = np.squeeze(df.predict(X))
#
#        return mutual_info_score(pred, target)
#    return 0  # 1 * log 1 = 0


#def log_likelihood(A, X, y, decision_function, n_random_iter=20, eps_norm=1e-8):
#    target = np.squeeze(y)
#    target_type = type_of_target(target)
#
#    if target_type not in ['binary', 'multiclass']:
#        raise Exception(target_type, 'not supported.')
#
#    lencoder = LabelEncoder()
#    decision_function = clone(decision_function)
#
#    y_arange = np.arange(len(np.squeeze(y)))
#    y_labels = lencoder.fit_transform(y)
#    ll = 0
#
#    if A:
#        X = X[:, A]
#
#        decision_function.fit(X, y)
#        y_pred = decision_function.predict_proba(X)
#        ll = np.sum(np.log(y_pred[y_arange, np.squeeze(y_labels)] + eps_norm))
#    else:
#
#        lls = list()
#        for i in range(n_random_iter):
#            y_pred = np.random.beta(1/2, 1/2, size=len(y))
#            lls.append(np.sum(np.log(y_pred + eps_norm)))
#        ll = np.mean(lls)
#
#    return ll


def log_likelihood_target(A, X_train, X_test, y_train, y_test, df, n_random_iter=20, eps_norm=1e-8, return_fitted:bool = False):
    target = np.squeeze(y_train)
    target_type = type_of_target(target)

    if target_type not in ['binary', 'multiclass']:
        raise Exception(target_type, 'not supported.')

    decision_function = clone(df)
    ll = 0

    y_test = np.copy(y_test)

    if A:
        X_train_m = make_features_matrix(X_train, A)
        X_test_m = make_features_matrix(X_test, A)

        decision_function.fit(X_train_m, y_train)
        y_pred = decision_function.predict_proba(X_test_m)

        # map test values to indices to calc log likelihood
        classes_ = get_model_classification_order(decision_function)
        y_test, y_pred = fix_target(classes_, y_test, y_pred)

        ll = np.sum(np.log(y_pred[np.arange(y_test.size), np.squeeze(y_test)] + eps_norm))
    else:

        lls = list()
        for i in range(n_random_iter):
            y_pred = np.random.beta(1/2, 1/2, size=len(y_test))
            lls.append(np.sum(np.log(y_pred + eps_norm)))
        ll = np.mean(lls)

    if return_fitted:
        return decision_function, ll

    del decision_function

    return ll


def likelihood_target(A, X_train, X_test, y_train, y_test, decision_function, n_random_iter=20, eps_norm=1e-8):
    return np.exp(log_likelihood_target(A, X_train, X_test, y_train, y_test, decision_function, n_random_iter, eps_norm))


def log_likelihood_bic(A, X_train, X_test, y_train, y_test, decision_function, n_random_iter=20, eps_norm=1e-8):
    df, ll = log_likelihood_target(A, X_train, X_test, y_train, y_test, decision_function, n_random_iter, eps_norm,
                                   return_fitted=True)
    n_coefs = get_model_n_coefs(df)
    del df

    return 2*ll - np.log(len(y_test)) * n_coefs


def log_likelihood_aic(A, X_train, X_test, y_train, y_test, decision_function, n_random_iter=20, eps_norm=1e-8):
    df, ll = log_likelihood_target(A, X_train, X_test, y_train, y_test, decision_function, n_random_iter, eps_norm,
                                   return_fitted=True)
    n_coefs = get_model_n_coefs(df)
    del df

    return 2*ll - 2 * n_coefs


def log_likelihood_aicc(A, X_train, X_test, y_train, y_test, decision_function, n_random_iter=20, eps_norm=1e-8):
    df, ll = log_likelihood_target(A, X_train, X_test, y_train, y_test, decision_function, n_random_iter, eps_norm,
                                   return_fitted=True)
    n_coefs = get_model_n_coefs(df)
    del df

    return 2*ll - 2 * n_coefs - float(2 * (n_coefs ** 2) + 2 * n_coefs) / float(len(y_test) - n_coefs - 1)

