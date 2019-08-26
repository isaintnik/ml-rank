from functools import partial

from .subset import *
from .target import *


def mutual_information_regularized_score(A, X_f, X_t, y, decision_function, _lambda) -> float:
    """
    score which is sensitive to features cross correlations in informational meaning
    :param A: subset
    :param X_f: raw features
    :param X_t: dichtomized features
    :param y: target
    :param decision_function: decision function
    :param _lambda: coefficient for regularization
    :return:
    """
    if not A:
        return 0

    return mutual_information_classification(A=A, X=X_f, y=y, decision_function=decision_function) - \
        _lambda * informational_regularization_classification(A=A, X_f=X_f, X_t=X_t, decision_function=decision_function)


def mutual_information_regularized_score_penalized(A, X_f, X_t, y, decision_function, _lambda, _gamma) -> float:
    """
    score which is sensitive to uninformational feature inclusion
    :param A: subset
    :param X_f: raw features
    :param X_t: dichtomized features
    :param y: target
    :param decision_function: decision function
    :param _lambda: coefficient for regularization
    :param _gamma: coefficient for uninformative feature penalty
    :return:
    """
    if not A:
        return 0

    return mutual_information_classification(A=A, X=X_f, y=y, decision_function=decision_function) - \
        float(_lambda) * informational_regularization_classification(A=A, X_f=X_f, X_t=X_t, decision_function=decision_function) - \
        float(_gamma) * len(A) / X_f.shape[1]


def mutual_info_bic():
    pass


def mutual_info_aic():
    pass
