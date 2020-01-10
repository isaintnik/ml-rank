from joblib import Parallel, delayed
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.datasets import make_blobs

from guppy import hpy; h=hpy()

import objgraph


def fit_model():
    X, y = make_blobs(n_samples=1000, n_features=10, centers=10)
    mdl = LGBMClassifier(
        boosting_type='gbdt',
        learning_rate=0.05,
        num_iterations=1200,
        max_depth=5,
        n_estimators=1000,
        verbose=-1,
        num_leaves=2 ** 5,
        silent=True,
        n_jobs=1
    )
    mdl.fit(X, y)


if __name__ == '__main__':
    before = h.heap()

    for j in range(100):
        results = Parallel(n_jobs=3)(
            delayed(fit_model)()
            for i in range(j)
        )

        del results

        objgraph.show_most_common_types(25)
        #print((h.heap() - before))
        #print('-'*100)
        #print('-' * 100)
        #print('-' * 100)

    #print(h.heap() - before)

    #for i in range(10):
    #    X, y = make_blobs(n_samples=1000, n_features=10, centers=10)
    #
    #    mdl = LGBMClassifier(
    #        boosting_type='gbdt',
    #        learning_rate=0.05,
    #        num_iterations=1200,
    #        max_depth=5,
    #        n_estimators=1000,
    #        verbose=-1,
    #        num_leaves=2 ** 5,
    #        silent=True,
    #        n_jobs=1
    #    )
    #
    #    mdl.fit(X, y)
    #
    #    print(h.heap() - before)