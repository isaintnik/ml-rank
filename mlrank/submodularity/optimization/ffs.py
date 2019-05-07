import numpy as np

from sklearn.base import clone

from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target

from mlrank.preprocessing.dichtomizer import MaxentropyMedianDichtomizationTransformer, map_continious_names
from mlrank.submodularity.optimization.optimizer import SubmodularOptimizer


class ForwardFeatureSelection(SubmodularOptimizer):
    def __init__(self, decision_function, score_function, train_share:float, n_cv_ffs:int, n_features: int, n_bins: 4):
        """
        :param decision_function:
        :param score_function:
        :param n_cv:
        :param train_share:
        :param n_cv_ffs:
        :param n_features:
        :param n_bins: only used for continuous targets
        """
        super().__init__()

        self.decision_function = clone(decision_function)
        self.score_function = score_function
        self.n_features = n_features
        self.n_cv_ffs = n_cv_ffs
        self.n_bins = n_bins
        self.train_share = train_share

        self.seeds = [(42 + i) for i in range(self.n_cv_ffs)]

    def select(self, X, y) -> list:
        df = clone(self.decision_function)

        subset = list()

        for i in range(self.n_features):
            feature_scores = list()

            for i in range(X.shape[1]):
                if i in subset:
                    feature_scores.append(0)
                    continue

                X_s = X[:, subset + [i]]
                y = np.squeeze(y)

                scores = list()

                for i in range(self.n_cv_ffs):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_s, y, random_state=self.seeds[i], shuffle=True, test_size = 1 - self.train_share
                    )

                    model = clone(df)

                    if type_of_target(y_train) == 'continuous':
                        dichtomizer = MaxentropyMedianDichtomizationTransformer(self.n_bins)
                        dichtomizer.fit(y_train.reshape(-1, 1))
                        train_target = dichtomizer.transform(y_train.reshape(-1, 1))
                        model.fit(X_train, train_target)

                        r_d = np.squeeze(dichtomizer.transform(y_test.reshape(-1, 1)))
                        p_d = model.predict(X_test)

                        scores.append(mutual_info_score(p_d, r_d))
                    else:
                        model.fit(X_train, y_train)
                        scores.append(mutual_info_score(model.predict(X_test), y_test))

                feature_scores.append(np.mean(scores))
            subset.append(np.atleast_1d(np.squeeze(np.argmax(feature_scores)))[0])

        return subset
