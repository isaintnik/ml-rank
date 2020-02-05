import concurrent.futures

import joblib
import os
import json
import requests
import numpy as np
import time

from joblib import Parallel, delayed
from sklearn import clone

from mlrank.utils import split_dataset
from .ffs import ForwardFeatureSelection


async def eval_new_feature(
        subset: list,
        new_feature: str,
        X_f: dict,
        X_t: dict,
        y,
        n_cv_ffs: int,
        n_jobs: int,
        seeds: list,
        train_share: float,
        score_function_components,
        decision_function
):
    A = subset + [new_feature]
    likelihoods = list()
    if n_jobs > 1:
        print(n_jobs)
        likelihoods = Parallel(n_jobs=n_jobs)(
            delayed(score_function_components)(
                A=A,
                X_f=result['train']['transformed'],
                X_f_test=result['test']['transformed'],
                X_t=result['train']['plain'],
                X_t_test=result['test']['plain'],
                y=result['train']['target'],
                y_test=result['test']['target'],
                decision_function=decision_function
            )
            for result in (
                split_dataset(X_t, X_f, y, seeds[i], 1 - train_share)
                for i in range(n_cv_ffs)
            )
        )
    else:
        for i in range(n_cv_ffs):
            result = split_dataset(X_t, X_f, y, seeds[i], 1 - train_share)

            likelihoods.append(score_function_components(
                A=A,
                X_f=result['train']['transformed'],
                X_f_test=result['test']['transformed'],
                X_t=result['train']['plain'],
                X_t_test=result['test']['plain'],
                y=result['train']['target'],
                y_test=result['test']['target'],
                decision_function=decision_function
            ))

    return {k: float(np.mean([l[k] for l in likelihoods]))
            for k in likelihoods[0].keys()}


class ForwardFeatureSelectionCompositeClient(ForwardFeatureSelection):
    def __init__(self,
                 server_clc: str,
                 port_clc: str,

                 server_res: str,
                 port_res: str,

                 decision_function: str,
                 score_function,
                 train_share: float = 1.0,
                 n_bins: int = 4,
                 n_features: int = -1,
                 n_cv_ffs: int = 1,
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
            n_jobs = -1
        )

        self.server_clc = server_clc
        self.port_clc = port_clc

        self.server_res = server_res
        self.port_res = port_res

        self.feature_names = None

        self.score_function = score_function

    def evaluate_new_feature(self, prev_subset, new_feature, X_f, X_t, y):
        pass

    @staticmethod
    def run_async_worker(host, port, values, sample_path):
        with open(sample_path, 'rb') as f:
            r = requests.post(f'http://{host}:{port}/', data=values, files={'sample': f})

            if r.status_code == 200:
                return json.loads(r.text)
        return None

    def evaluate_from_external_service(self, sample_path: str, rndkey: int, prev_subset: list):
        free_features = set(self.feature_names).difference(prev_subset)

        result = dict()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(free_features)) as executor:
            future_to_feature = dict()
            for i in free_features:
                values = {
                    'storage_host': self.server_res,
                    'storage_port': self.port_res,
                    'key': rndkey,
                    'feature': i,
                    'subset': ','.join(prev_subset),
                    'decision_function': self.decision_function,
                    'n_cv_ffs': self.n_cv_ffs,
                    'ffs_train_share': self.train_share,
                    'seeds': ','.join(map(str, self.seeds))
                }

                time.sleep(0.05)

                future_to_feature[
                    executor.submit(
                        ForwardFeatureSelectionCompositeClient.run_async_worker,
                        self.server_clc, self.port_clc, values, sample_path
                    )
                ] = i

            for future in concurrent.futures.as_completed(future_to_feature):
                feature = future_to_feature[future]
                try:
                    result[feature] = future.result()
                except Exception as ex:
                    print(str(ex))
                    result[feature] = None

        if any(map(lambda x: x is None, result.values())):
            raise Exception(f'some features failed to process.')

        return result

    def evaluate_feature_score(self, ll_vals_prev, ll_vals_cur) -> float:
        return self.score_function(ll_vals_prev, ll_vals_cur)

    def select(self, X_plain: dict, X_transformed: dict, y: np.array, continuous_feature_list: list) -> list:
        self.feature_names = list(X_plain.keys())

        X_f, X_t, y = self.get_dichotomized(X_plain, X_transformed, y, continuous_feature_list)

        os.makedirs('./tmp/', exist_ok=True)
        joblib.dump({
            'X_f': X_f,
            'X_t': X_t,
            'y': y,
        }, './tmp/sample.bin')

        if self.n_features == -1:
            self.n_features = len(X_plain.keys())

        subset = list()
        prev_top_score = -np.inf
        values_prev = None
        for _ in self.feature_names:
            eval_key = int(np.random.randint(0, 10000000))
            result = self.evaluate_from_external_service(sample_path='./tmp/sample.bin', rndkey=eval_key, prev_subset=subset)

            feature_scores = list()
            ordered_feature_names = list(result.keys())
            for j in ordered_feature_names:
                feature_scores.append(self.evaluate_feature_score(values_prev, result[j]))

            top_feature = int(np.argmax(feature_scores))  # np.atleast_1d(np.squeeze(np.argmax(feature_scores)))[0]

            if np.max(feature_scores) > prev_top_score:
                subset.append(ordered_feature_names[top_feature])
                values_prev = result[ordered_feature_names[top_feature]]
                prev_top_score = np.max(feature_scores)

                self.logs.append({
                    'subset': np.copy(subset).tolist(),
                    'score': np.max(feature_scores)
                })
            else:
                break

        return subset

    def get_logs(self):
        return self.logs