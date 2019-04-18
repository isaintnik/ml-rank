import numpy as np
from warnings import warn

from sklearn.metrics import mutual_info_score
from sklearn.utils.multiclass import type_of_target
from sklearn.base import clone

from mlrank.preprocessing.dichtomizer import MaxentropyMedianDichtomizationTransformer, map_continious_names


def mutual_information(A, X, y, decision_function, n_bins=4):
    # TODO: could be precalculated
    df = clone(decision_function)
    df.fit(X[:, A], np.squeeze(y))

    pred = np.squeeze(df.predict(X[:, A]))
    target = np.squeeze(y)

    if type_of_target(y) == 'continuous':
        dichtomizer = MaxentropyMedianDichtomizationTransformer(n_bins)
        dichtomizer.fit(y.reshape(-1, 1))

        pred = np.squeeze(dichtomizer.transform_ordered(pred.reshape(-1, 1)))
        target = np.squeeze(dichtomizer.transform_ordered(y.reshape(-1, 1)))


    labels = np.unique(target).tolist()

    return mutual_info_score(
        map_continious_names(pred, continious_labels=labels),
        map_continious_names(target, continious_labels=labels)
    )
