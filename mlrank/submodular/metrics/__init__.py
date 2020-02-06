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


def get_log_likelihood_regularized_score_balanced_components(A, X_f, X_f_test, X_t, X_t_test, y, y_test, decision_function) -> dict:
    if not hasattr(decision_function, 'predict_proba'):
        raise Exception('decision function should have predict_proba')

    ll = log_likelihood_target(A, X_f, X_f_test, y, y_test, decision_function, 20)
    llcf = log_likelihood_cross_features(A, X_f, X_f_test, X_t, X_t_test, decision_function, 20)

    return {
        "ll": ll,
        "llcf": llcf
    }


# argmin problem
def log_likelihood_regularized_score_multiplicative_balanced(components_prev, components_cur, _lambda: float) -> float:
    if components_prev is None:
        return components_cur['ll']# - _lambda * components_cur['llcf']

    #return (components_cur['ll'] - components_prev['ll']) - _lambda * (components_cur['llcf'] - components_prev['llcf'])
    return (components_cur['ll'] - components_prev['ll']) * (
        1 - _lambda * (components_cur['llcf'] - components_prev['llcf'])
    )


# argmax problem
def likelihood_regularized_score_val(A, X_f, X_f_test, X_t, X_t_test, y, y_test, decision_function, _lambda) -> np.float128:
    val = log_likelihood_regularized_score_val(A, X_f, X_f_test, X_t, X_t_test, y, y_test, decision_function, _lambda)
    return np.exp(np.array(val, dtype=np.float128))


# argmax problem
def bic_criterion(A, X_f, X_f_test, X_t, X_t_test, y, y_test, decision_function):
    return log_likelihood_bic(A, X_f, X_f_test, y, y_test, decision_function)


# argmax problem
def aic_criterion(A, X_f, X_f_test, X_t, X_t_test, y, y_test, decision_function):
    return log_likelihood_aic(A, X_f, X_f_test, y, y_test, decision_function)


# argmax problem
def aicc_criterion(A, X_f, X_f_test, X_t, X_t_test, y, y_test, decision_function):
    return log_likelihood_aicc(A, X_f, X_f_test, y, y_test, decision_function)
