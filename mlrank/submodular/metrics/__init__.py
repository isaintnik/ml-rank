from functools import partial

from .subset import *
from .target import *


def log_likelihood_regularized_score_val(A, X_f, X_f_test, X_t, X_t_test, y, y_test, decision_function, _lambda) -> float:
    if not hasattr(decision_function, 'predict_proba'):
        raise Exception('decision function should have predict_proba')

    ll = log_likelihood_target(A, X_f, X_f_test, y, y_test, decision_function, 20)
    if _lambda > 0:
        llcf = log_likelihood_cross_features(A, X_f, X_f_test, X_t, X_t_test, decision_function, 20)
    else:
        llcf = 0
    return ll - _lambda * llcf


def bic_regularized(A, X_f, X_f_test, X_t, X_t_test, y, y_test, decision_function):
    return log_likelihood_bic(A, X_f, X_f_test, y, y_test, decision_function)


def aic_regularized(A, X_f, X_f_test, X_t, X_t_test, y, y_test, decision_function):
    return log_likelihood_aic(A, X_f, X_f_test, X_t, X_t_test, y, y_test, decision_function)


def base_score(A, X_f, X_f_test, X_t, X_t_test, y, y_test, decision_function, score) -> float:
    pass