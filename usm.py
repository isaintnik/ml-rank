from functools import partial

import numpy as np
from sklearn.linear_model import LogisticRegression

from mlrank.preprocessing.dichtomizer import dichotomize_vector
from mlrank.synth.linear import LinearProblemGenerator
from mlrank.synth.nonlinear import NonlinearProblemGenerator
from mlrank.submodular.optimization.ffs import (
    ForwardFeatureSelectionExtended,
    ForwardFeatureSelectionClassic
)
from mlrank.submodular.optimization.multilinear import (
    MultilinearUSMExtended,
    MultilinearUSMClassic
)
from mlrank.submodular.metrics import (
    mutual_information_regularized_score_penalized,
    log_likelihood_regularized_score,
    log_likelihood_regularized_score_val
)
from mlrank.submodular.metrics.target import log_likelihood
from sklearn.metrics import mutual_info_score

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    np.random.seed(42)
    n=200
    #data = LinearProblemGenerator.make_mc_uniform(n, np.array([2, 5, -3]), 2, 8)#(500, 10, 10, 5)
    #data = NonlinearProblemGenerator.make_nonlinear_relations_problem()
    data = LinearProblemGenerator.make_normal_normal(n, coefs=np.array([2, 5, -3, -2]), n_junk=8)  # (500, 10, 10, 5)

    X = np.hstack(data['features'])
    y = data['target']

    print(data['mask'])

    decision_function = LogisticRegression(multi_class='ovr', solver='liblinear', C=10000)
    #score_function = partial(mutual_information_regularized_score_penalized, _lambda = 1.0, _gamma = 0.3)
    #print(log_likelihood([0, 1], X, dichtomize_vector(y, 8), decision_function))

    for i in [2, 4, 8]:
        for j in np.linspace(0, 1.0, 1):
            print(i, j)
            mplik = -n * (np.log(1./i))
            coef = 2 * i / mplik
            print(coef)
            score_function = partial(log_likelihood_regularized_score_val, _lambda=j)

            usm = ForwardFeatureSelectionExtended(
                decision_function,
                score_function,
                n_bins=i,
                train_share=0.8,
                n_cv_ffs=6,
                #me_eps=.15,
                #threshold=.5
            )

            print(usm.select(X, y))
            #print(usm.get_logs()[-1]['score'])

    print('-' * 100)
    print('-' * 100)

    #for i in [2, 4, 8]:
    #    ums = MultilinearUSMClassic(
    #        decision_function,
    #        mutual_info_score,
    #        n_bins=i,
    #        me_eps=.1,
    #        threshold=.5,
    #        n_cv=1,
    #        train_share=.8
    #    )
    #
    #    print(ums.select(X, y))
