import numpy as np
from warnings import warn

from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target
from sklearn.base import clone

from mlrank.preprocessing.dichtomizer import MaxentropyMedianDichtomizationTransformer, map_continuous_names


#def mutual_information(A, X, y, decision_function, n_bins=4, n_cv=1):
#    # TODO: could be precalculated
#    scores = list()
#
#    for i in range(n_cv):
#        df = clone(decision_function)
#        df.fit(X[:, A], np.squeeze(y))
#
#        pred = np.squeeze(df.predict(X[:, A]))
#        target = np.squeeze(y)
#
#        if type_of_target(y) == 'continuous':
#            dichtomizer = MaxentropyMedianDichtomizationTransformer(n_bins)
#            dichtomizer.fit(y.reshape(-1, 1))
#
#            pred = np.squeeze(dichtomizer.transform_ordered(pred.reshape(-1, 1)))
#            target = np.squeeze(dichtomizer.transform_ordered(y.reshape(-1, 1)))
#
#        labels = np.unique(target).tolist()
#
#        scores.append(mutual_info_score(
#            map_continious_names(pred, continious_labels=labels),
#            map_continious_names(target, continious_labels=labels)
#        ))
#
#    return np.mean(scores)


#def mutual_information_classification(A, X, y, decision_function, n_bins=4, n_cv=1, test_share=.2):
#    # TODO: could be precalculated
#    scores = list()
#
#    X = X[:, A]
#
#    for i in range(n_cv):
#        df = clone(decision_function)
#
#        X_train, X_test, y_train, y_test = train_test_split(X, y, test_share=.2)
#
#        if type_of_target(y) == 'continuous':
#            dichtomizer = MaxentropyMedianDichtomizationTransformer(n_bins)
#            dichtomizer.fit(y_train.reshape(-1, 1))
#
#            target = np.squeeze(dichtomizer.transform_ordered(y_train.reshape(-1, 1)))
#            df.fit(X_train, target)
#            pred = np.squeeze(df.predict(X_test))
#        else:
#            df.fit(X_train, np.squeeze(y_train))
#
#            target = np.squeeze(y_test)
#            pred = np.squeeze(df.predict(X_test))
#
#        labels = np.unique(target).tolist()
#
#        scores.append(mutual_info_score(
#            map_continious_names(pred, continious_labels=labels),
#            map_continious_names(target, continious_labels=labels)
#        ))
#
#    return np.mean(scores)


def mutual_information_classification(A, X, y, decision_function):
    X = X[:, A]

    target = np.squeeze(y)

    if np.unique(y).shape[0] > 1:
        df = clone(decision_function)
        df.fit(X, target)

        pred = np.squeeze(df.predict(X))

        return mutual_info_score(pred, target)
    return mutual_info_score(target, target)


def mutual_information_classification_cv(A, X, y, decision_function, n_bins=4):
    X = X[:, A]

    df = clone(decision_function)

    if type_of_target(y) == 'continuous':
        dichtomizer = MaxentropyMedianDichtomizationTransformer(n_bins)
        dichtomizer.fit(y.reshape(-1, 1))

        target = np.squeeze(dichtomizer.transform(y.reshape(-1, 1)))
        df.fit(X, target)
        pred = np.squeeze(df.predict(X))
    else:
        df.fit(X, np.squeeze(y))

        target = np.squeeze(y)
        pred = np.squeeze(df.predict(X))

    labels = np.unique(target).tolist()

    return mutual_info_score(
        map_continuous_names(pred, continuous_labels=labels),
        map_continuous_names(target, continuous_labels=labels)
    )