import numpy as np
from warnings import warn

from sklearn.metrics import mutual_info_score
from sklearn.utils.multiclass import type_of_target
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder

from mlrank.preprocessing.dichtomizer import MaxentropyMedianDichtomizationTransformer, map_continious_names

from pyitlib.discrete_random_variable import entropy_joint


def mutual_information_normalized(features, target, decision_function, n_bins=4):
    if type_of_target(target) in ['binary', 'multiclass']:
        raise Exception('mutual information is calculated only for continious variables at the moment')

    dichtomizer = MaxentropyMedianDichtomizationTransformer(n_bins)
    dichtomizer.fit(target.reshape(-1, 1))

    df = clone(decision_function)
    df.fit(features, target)
    pred = df.predict(features)

    pred = np.squeeze(dichtomizer.transform_ordered(pred.reshape(-1, 1)))
    target = np.squeeze(dichtomizer.transform_ordered(target.reshape(-1, 1)))

    labels = np.unique(target).tolist()

    return mutual_info_score(
        map_continious_names(pred, continious_labels=labels),
        map_continious_names(target, continious_labels=labels)
    ) / mutual_info_score(
        map_continious_names(target, continious_labels=labels),
        map_continious_names(target, continious_labels=labels)
    )
