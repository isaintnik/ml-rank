from functools import partial

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from mlrank.preprocessing.dichtomizer import dichotomize_vector
from mlrank.synth.linear import LinearProblemGenerator
from sklearn.metrics import mutual_info_score
from mlrank.submodular.optimization.ffs import (
    ForwardFeatureSelectionClassic,
    ForwardFeatureSelectionExtended
)

from mlrank.submodular.metrics import mutual_information_regularized_score_penalized


if __name__ == '__main__':
    np.random.seed(42)
    data = LinearProblemGenerator.make_mc_uniform(300, np.array([.2, 5, -3]), 2, 5)#(500, 10, 10, 5)

    X = np.hstack(data['features'])
    y = data['target']

    print(data['mask'])

    decision_function = LogisticRegression(multi_class='ovr', solver='liblinear')
    score_function = partial(mutual_information_regularized_score_penalized, _lambda = 0.5, _gamma = 0.3)#, _lambda = 0.5, _gamma = 0.1)

    print('new metric')

    #for i in [2, 4, 8]:
    #    ffs = ForwardFeatureSelectionExtended(
    #        decision_function,
    #        score_function,
    #        n_bins=i
    #    )
    #
    #    print(ffs.select(X, y))
    #    print(ffs.get_logs())
    #
    #print('classic mutual information')

    for i in [2, 4, 8]:
        ffs = ForwardFeatureSelectionClassic(
            decision_function,
            mutual_info_score,
            n_bins=i,
            train_share=0.6,
            n_cv_ffs=1,
            n_features=5
        )

        print(ffs.select(X, y))
        print(ffs.get_logs())