import numpy as np
from sklearn import clone

from sklearn.model_selection import train_test_split


class USMBenchmark(object):
    def __init__(self, decision_funciton, score_function, treshold:float, n_cv:int, train_share:float):
        self.treshold = treshold
        self.decision_function = clone(decision_funciton)
        self.score_function = score_function
        self.n_cv = n_cv
        self.train_share = train_share

        self.seeds = [(42 + i) for i in range(self.n_cv)]

    def evaluate(self, X, y, features_param) -> np.ndarray:
        features = np.array(features_param)

        X = X[:, np.array(features >= self.treshold)]
        y = np.squeeze(y)

        scores = list()

        for i in range(self.n_cv):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, random_state=self.seeds[i], shuffle=True, test_size = 1 - self.train_share
            )

            model = clone(self.decision_function)

            model.fit(X_train, y_train)
            scores.append(self.score_function(model.predict(X_test), y_test))

        return np.mean(scores)
