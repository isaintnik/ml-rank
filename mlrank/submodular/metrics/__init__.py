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

    a = mutual_information_classification(A=A, X=X_f, y=y, decision_function=decision_function)
    if _lambda > 0.:
        b = float(_lambda) * informational_regularization_classification(A=A, X_f=X_f, X_t=X_t,
                                                                         decision_function=decision_function)
    else:
        b = 0

    return a - b


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

    a = mutual_information_classification(A=A, X=X_f, y=y, decision_function=decision_function)
    if _lambda > 0.:
        b = float(_lambda) * informational_regularization_classification(A=A, X_f=X_f, X_t=X_t, decision_function=decision_function)
    else:
        b = 0

    c = float(_gamma) * float(len(A)) / X_f.shape[1]

    return a - b# - c


def log_likelihood_regularized_score_bic(A, X_f, X_t, y, decision_function, _lambda, _gamma) -> float:
    if not hasattr(decision_function, 'predict_proba'):
        raise Exception('decision function should have predict_proba')

    ll = log_likelihood(A, X_f, y, decision_function, 20)
    llcf = log_likelihood_cross_features(A, X_f, X_t, decision_function, 20)
    bic = np.log(X_f.shape[0]) * len(A)

    #print(A, ll, llcf, bic)

    return ll - _lambda * llcf# - _gamma * bic


def mutual_info_bic():
    pass


def mutual_info_aic():
    pass
