import numpy as np

from sklearn.metrics import mutual_info_score
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
