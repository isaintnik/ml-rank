import numpy as np

from sklearn.base import clone

from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target

from mlrank.preprocessing.dichtomizer import MaxentropyMedianDichtomizationTransformer, DichtomizationIssue
from mlrank.submodular.metrics import mutual_information_regularized_score_penalized
from mlrank.submodular.optimization.optimizer import SubmodularOptimizer

from functools import partial


class ForwardFeatureSelection(SubmodularOptimizer):
    def __init__(self,
                 decision_function,
                 score_function,
                 train_share: float = 1.0,
                 n_bins: int = 4,
                 n_features: int = -1,
                 n_cv_ffs: int = 1
                 ):
        """
        Perform greedy algorithm of feature selection ~ O(n_features ** 2)
        :param decision_function: decision function to be evaluated
        :param score_function: score function for submodular optimization
        :param train_share: share of data to be trained on
        :param n_cv_ffs: number of CV's, 1 = evaluate on training set
        :param n_bins: only used for continuous targets
        """
        super().__init__()

        self.decision_function = clone(decision_function)
        self.score_function = score_function
        self.n_cv_ffs = n_cv_ffs
        self.n_bins = n_bins
        self.train_share = train_share
        self.n_features = n_features
        self.logs = None

        self.seeds = [(42 + i) for i in range(self.n_cv_ffs)]

    def select(self, X, y) -> list:
        raise NotImplementedError()

    def get_logs(self):
        return self.logs


class ForwardFeatureSelectionClassic(ForwardFeatureSelection):
    def __init__(self,
                 decision_function,
                 score_function,
                 train_share: float = 1.0,
                 n_bins: int = 4,
                 n_features: int = -1,
                 n_cv_ffs: int = 1
        ):
        """
        Perform greedy algorithm of feature selection ~ O(n_features ** 2)
        :param decision_function: decision function to be evaluated
        :param score_function: score function for submodular optimization
        :param train_share: share of data to be trained on
        :param n_cv_ffs: number of CV's, 1 = evaluate on training set
        :param n_bins: only used for continuous targets
        """
        super().__init__(
            decision_function,
            score_function,
            train_share,
            n_bins,
            n_features,
            n_cv_ffs
        )

    def _evaluate_model(self, X_train, y_train, X_test, y_test, model) -> float:
        #if type_of_target(y_train) == 'continuous':
        #    #dichtomizer = MaxentropyMedianDichtomizationTransformer(self.n_bins)
        #    #dichtomizer.fit(y_train.reshape(-1, 1))
        #    #train_target = dichtomizer.transform(y_train.reshape(-1, 1))
        #    #model.fit(X_train, train_target)
        #    #
        #    #r_d = np.squeeze(dichtomizer.transform(y_test.reshape(-1, 1)))
        #    #p_d = model.predict(X_test)
        #    #
        #    #return self.score_function(p_d, r_d)
        #
        #    raise Exception('Target is not discrete.')
        #else:

        model.fit(X_train, y_train)
        return self.score_function(model.predict(X_test), y_test)

    def _evaluate_new_feature(self, prev_subset, new_feature, X, y) -> float:
        X_s = X[:, prev_subset + [new_feature]]
        y = np.squeeze(y)

        scores = list()

        if self.n_cv_ffs > 1:
            for i in range(self.n_cv_ffs):
                X_train, X_test, y_train, y_test = train_test_split(
                    X_s, y, random_state=self.seeds[i], shuffle=True, test_size= 1 - self.train_share
                )

                model = clone(self.decision_function)
                scores.append(self._evaluate_model(X_train, y_train, X_test, y_test, model))

            return float(np.mean(scores))
        else:
            model = clone(self.decision_function)
            return self._evaluate_model(X_s, y, X_s, y, model)

    def select(self, X, y) -> list:
        try:
            y = self.dichtomize_target(y, self.n_bins)
        except Exception as e:
            print(e)
            raise DichtomizationIssue(self.n_bins)

        subset = list()
        subset_logs = list()
        if self.n_features == -1:
            self.n_features = X.shape[1]
        self.logs = list()
        prev_top_score = -np.inf

        for i in range(self.n_features):
            feature_scores = list()

            for i in range(X.shape[1]):
                if i in subset_logs:
                    feature_scores.append(-np.inf)
                    continue

                feature_scores.append(self._evaluate_new_feature(subset, i, X, y))

            top_feature = np.atleast_1d(np.squeeze(np.argmax(feature_scores)))[0]

            # if top score of new feature is not significant - ignore this and further features
            # however, if n_features has been specified - proceed
            if np.max(feature_scores) > prev_top_score or self.n_features < X.shape[1]:
                subset.append(top_feature)
                prev_top_score = np.max(feature_scores)
            else:
                break

            subset_logs.append(top_feature)

            self.logs.append({
                'subset': np.copy(subset_logs).tolist(),
                'score': np.max(feature_scores)
            })

        return subset

    def get_logs(self):
        return self.logs


class ForwardFeatureSelectionExtended(ForwardFeatureSelection):
    def __init__(self,
                 decision_function,
                 score_function,
                 train_share: float = 1.0,
                 n_bins: int = 4,
                 n_features: int = -1,
                 n_cv_ffs: int = 1
                 ):
        """
        Perform greedy algorithm of feature selection ~ O(n_features ** 2)
        :param decision_function: decision function to be evaluated
        :param score_function: score function for submodular optimization
        :param train_share: share of data to be trained on
        :param n_cv_ffs: number of CV's, 1 = evaluate on training set
        :param n_bins: only used for continuous targets
        """
        super().__init__(
            decision_function,
            score_function,
            train_share,
            n_bins,
            n_features,
            n_cv_ffs
        )

    def _evaluate_new_feature(self, prev_subset, new_feature, X_f, X_t, y) -> float:
        A = prev_subset + [new_feature]

        scores = list()
        for i in range(self.n_cv_ffs):
            X_f_train, X_f_test, X_t_train, X_t_test, y_train, y_test = train_test_split(
                X_f, X_t, y, random_state=self.seeds[i], shuffle=True, test_size=1 - self.train_share
            )

            scores.append(
                #self.score_function(A=A, X_f=X_f, X_t=X_t, y=y, decision_function=self.decision_function)
                self.score_function(
                    A=A, X_f=X_f_train, X_f_test=X_f_test,
                    X_t=X_t_train, X_t_test=X_t_test,
                    y=y_train, y_test=y_test,
                    decision_function=self.decision_function
                )
            )

        return np.mean(scores)

    def select(self, X, y) -> list:
        try:
            X_f = X
            X_t = self.dichtomize_features(X, self.n_bins)
            y = self.dichtomize_target(y, self.n_bins)
        except Exception as e:
            print(e)
            raise DichtomizationIssue(self.n_bins)

        subset = list()
        subset_logs = list()
        if self.n_features == -1:
            self.n_features = X.shape[1]
        self.logs = list()

        prev_top_score = -np.inf

        for i in range(self.n_features):
            feature_scores = list()

            for i in range(self.n_features):
                if i in subset_logs:
                    feature_scores.append(-np.inf)
                    continue

                feature_scores.append(self._evaluate_new_feature(subset, i, X_f, X_t, y))

            top_feature = np.argmax(feature_scores)#np.atleast_1d(np.squeeze(np.argmax(feature_scores)))[0]

            if np.max(feature_scores) > prev_top_score  or self.n_features < X.shape[1]:
                subset.append(top_feature)
                prev_top_score = np.max(feature_scores)
            else:
                break

            subset_logs.append(top_feature)

            self.logs.append({
                'subset': np.copy(subset_logs).tolist(),
                'score': np.max(feature_scores)
            })
        return subset
