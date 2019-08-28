from functools import partial

import numpy as np
from sklearn.linear_model import LogisticRegression

from mlrank.preprocessing.dichtomizer import dichtomize_vector
from mlrank.synth.linear import LinearProblemGenerator
from mlrank.submodular.optimization.usm import (
    MultilinearUSMExtended,
    MultilinearUSMClassic
)
from mlrank.submodular.metrics import (
    mutual_information_regularized_score_penalized,
    log_likelihood_regularized_score_bic
)
from mlrank.submodular.metrics.target import log_likelihood
from sklearn.metrics import mutual_info_score


if __name__ == '__main__':
    np.random.seed(42)
    data = LinearProblemGenerator.make_mc_uniform(300, np.array([2, 5, -3]), 2, 5)#(500, 10, 10, 5)
    #data = LinearProblemGenerator.make_normal_normal(300, coefs=np.array([2, 5, -3]), n_junk=4)  # (500, 10, 10, 5)

    X = np.hstack(data['features'])
    y = data['target']

    print(data['mask'])

    decision_function = LogisticRegression(multi_class='ovr', solver='liblinear')
    #score_function = partial(mutual_information_regularized_score_penalized, _lambda = 1.0, _gamma = 0.3)
    score_function = partial(log_likelihood_regularized_score_bic, _lambda=0.5, _gamma=1.0)
    #print(log_likelihood([0, 1], X, dichtomize_vector(y, 8), decision_function))

    #for i in [2, 4, 8]:
    #    ums = MultilinearUSMExtended(
    #        decision_function,
    #        score_function,
    #        n_bins=i,
    #        me_eps=.15,
    #        threshold=.5
    #    )
    #
    #    print(ums.select(X, y))
    #
    #print('-' * 100)
    #print('-' * 100)

    for i in [2, 4, 8]:
        ums = MultilinearUSMClassic(
            decision_function,
            mutual_info_score,
            n_bins=i,
            me_eps=.1,
            threshold=.5,
            n_cv=1,
            train_share=.8
        )

        print(ums.select(X, y))
