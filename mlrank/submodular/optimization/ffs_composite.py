import numpy as np
from joblib import Parallel, delayed
from sklearn import clone

from mlrank.utils import split_dataset
from .ffs import ForwardFeatureSelection


class ForwardFeatureSelectionComposite(ForwardFeatureSelection):
    def __init__(self,
                 decision_function,
                 score_function_components,
                 score_function,
                 train_share: float = 1.0,
                 n_bins: int = 4,
                 n_features: int = -1,
                 n_cv_ffs: int = 1,
                 n_jobs=-1
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
            train_share,
            n_bins,
            n_features,
            n_cv_ffs,
            n_jobs
        )

        self.score_function_components = score_function_components
        self.score_function = score_function

    def evaluate_new_feature(self, prev_subset, new_feature, X_f, X_t, y) -> dict:
        A = prev_subset + [new_feature]
        likelihoods = list()
        if self.n_jobs > 1:
            likelihoods = Parallel(n_jobs=self.n_jobs)(
                delayed(self.score_function_components)(
                    A=A,
                    X_f=result['train']['transformed'],
                    X_f_test=result['test']['transformed'],
                    X_t=result['train']['plain'],
                    X_t_test=result['test']['plain'],
                    y=result['train']['target'],
                    y_test=result['test']['target'],
                    decision_function=clone(self.decision_function)
                )
                for result in (
                    split_dataset(X_t, X_f, y, self.seeds[i], 1 - self.train_share)
                    for i in range(self.n_cv_ffs)
                )
            )
        else:
            for i in range(self.n_cv_ffs):
                result = split_dataset(X_t, X_f, y, self.seeds[i], 1 - self.train_share)

                likelihoods.append(self.score_function_components(
                    A=A,
                    X_f=result['train']['transformed'],
                    X_f_test=result['test']['transformed'],
                    X_t=result['train']['plain'],
                    X_t_test=result['test']['plain'],
                    y=result['train']['target'],
                    y_test=result['test']['target'],
                    decision_function=self.decision_function
                ))

        return {k: np.mean([l[k] for l in likelihoods]) for k in likelihoods[0].keys()}

    def evaluate_feature_score(self, ll_vals_prev, ll_vals_cur) -> float:
        return self.score_function(ll_vals_prev, ll_vals_cur)

    def select(self, X_plain: dict, X_transformed: dict, y: np.array, continuous_feature_list: list) -> list:
        X_f, X_t, y = self.get_dichotomized(X_plain, X_transformed, y, continuous_feature_list)

        subset = list()
        self.logs = list()

        feature_names = list(X_plain.keys())
        values_prev = None

        for i in range(len(feature_names)):
            feature_scores = list()

            values_cur_iteration = list()
            for j in feature_names:
                if j in subset:
                    values_cur_iteration.append(None)
                    feature_scores.append(-np.inf)
                else:
                    values_cur_iteration.append(self.evaluate_new_feature(subset, j, X_f, X_t, y))
                    feature_scores.append(self.evaluate_feature_score(values_prev, values_cur_iteration[-1]))

            top_feature = int(np.argmax(feature_scores))  # np.atleast_1d(np.squeeze(np.argmax(feature_scores)))[0]

            if np.max(feature_scores) > 0 or i == 0:
                subset.append(feature_names[top_feature])
                values_prev = values_cur_iteration[top_feature]

                self.logs.append({
                    'subset': np.copy(subset).tolist(),
                    'score': np.max(feature_scores)
                })
            else:
                break

        return subset

    def get_logs(self):
        return self.logs