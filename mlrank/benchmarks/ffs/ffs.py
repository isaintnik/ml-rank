import numpy as np
from sklearn import clone
from sklearn.metrics import mutual_info_score

from sklearn.model_selection import train_test_split


class FFSBenchmark(object):
    def __init__(self, decision_funciton, score_function, treshold:float, n_cv:int, train_share:float, n_cv_ffs:int):
        self.treshold = treshold
        self.decision_function = clone(decision_funciton)
        self.score_function = score_function
        self.n_cv = n_cv
        self.n_cv_ffs = n_cv_ffs
        self.train_share = train_share

        self.seeds = [(42 + i) for i in range(self.n_cv)]

    def evaluate(self, X, y, n_features) -> np.ndarray:
        subset = list()

        for i in range(n_features):
            feature_scores = list()

            for i in range(X.shape[1]):
                if i in subset:
                    feature_scores.append(0)
                    continue

                X_s = X[:, subset + [i]]
                y = np.squeeze(y)

                scores = list()

                model = clone(self.decision_function)
                model.fit(X_s, y)
                feature_scores.append(mutual_info_score(model.predict(X_s), y))

                for i in range(self.n_cv_ffs):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_s, y, random_state=self.seeds[i], shuffle=True, test_size = 1 - self.train_share
                    )

                    model = clone(self.decision_function)

                    model.fit(X_train, y_train)
                    scores.append(mutual_info_score(model.predict(X_test), y_test))
                feature_scores.append(np.mean(scores))
            subset.append(np.atleast_1d(np.squeeze(np.argmax(feature_scores)))[0])

        scores = list()

        for i in range(self.n_cv):
            X_train, X_test, y_train, y_test = train_test_split(
                X[:, subset], y, random_state=self.seeds[i], shuffle=True, test_size=1 - self.train_share
            )

            model = clone(self.decision_function)

            model.fit(X_train, y_train)
            scores.append(self.score_function(model.predict(X_test), y_test))

        return np.mean(scores)
