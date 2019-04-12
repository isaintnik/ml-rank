import numpy as np
from warnings import warn

from sklearn.metrics import mutual_info_score
from sklearn.utils.multiclass import type_of_target
from sklearn.base import clone

from mlrank.preprocessing.dichtomizer import MaxentropyMedianDichtomizationTransformer, map_continious_names


def mutual_information(A, X_c, y, decision_function, n_bins=4):
    if type_of_target(y) in ['binary', 'multiclass']:
        raise Exception('mutual information is calculated only for continious variables at the moment')

    # TODO: could be precalculated
    dichtomizer = MaxentropyMedianDichtomizationTransformer(n_bins)
    dichtomizer.fit(y.reshape(-1, 1))

    df = clone(decision_function)
    df.fit(X_c[:, A], y)
    pred = df.predict(X_c[:, A])

    pred = np.squeeze(dichtomizer.transform_ordered(pred.reshape(-1, 1)))
    # TODO: could be precalculated
    target = np.squeeze(dichtomizer.transform_ordered(y.reshape(-1, 1)))

    labels = np.unique(target).tolist()

    return mutual_info_score(
        map_continious_names(pred, continious_labels=labels),
        map_continious_names(target, continious_labels=labels)
    )
