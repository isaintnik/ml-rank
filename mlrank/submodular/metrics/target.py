import numpy as np

from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.utils.multiclass import type_of_target
from sklearn.base import clone


def mutual_information_classification(A, X, y, decision_function):
    X = X[:, A]

    target = np.squeeze(y)
    target_type = type_of_target(target)

    if target_type not in ['binary', 'multiclass']:
        raise Exception(target_type, 'not supported.')

    if np.unique(target).shape[0] > 1:
        df = clone(decision_function)
        df.fit(X, target)

        pred = np.squeeze(df.predict(X))

        return mutual_info_score(pred, target)
    return 0  # 1 * log 1 = 0


def log_likelihood(A, X, y, decision_function, n_random_iter=20, eps_norm=1e-8):
    target = np.squeeze(y)
    target_type = type_of_target(target)

    if target_type not in ['binary', 'multiclass']:
        raise Exception(target_type, 'not supported.')

    lencoder = LabelEncoder()
    decision_function = clone(decision_function)

    y_arange = np.arange(len(np.squeeze(y)))
    y_labels = lencoder.fit_transform(y)
    unique_y = np.unique(y)
    ll = 0

    if A:
        X = X[:, A]

        decision_function.fit(X, y)
        y_pred = decision_function.predict_proba(X)
        ll = np.sum(np.log(y_pred[y_arange, np.squeeze(y_labels)] + eps_norm))
    else:

        lls = list()
        for i in range(n_random_iter):
            y_pred = np.random.beta(1/2, 1/2, size=len(y))
            lls.append(np.sum(np.log(y_pred + eps_norm)))
        ll = np.mean(lls)

    return ll