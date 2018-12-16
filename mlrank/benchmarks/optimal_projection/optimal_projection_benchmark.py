import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats

from tqdm import tqdm
from copy import deepcopy

from .optimal_projection_algorithm import OPAlgorithm


class OptimalProjectionBenchmark(object):
    def __init__(self,
                 estimator,
                 projection_algorithm: OPAlgorithm,
                 n_holdout_validations: int,
                 n_min_projections: int = 1,
                 n_max_projections: int = None,
                 metric = accuracy_score):

        if n_min_projections < 1:
            raise Exception(f'[ERROR] invalid n_min_projections value {n_min_projections}')

        if n_min_projections > n_max_projections:
            raise Exception(f'[ERROR] n_min_projections > n_max_projections')

        self.estimator = estimator
        self.n_holdout_validations = n_holdout_validations
        self.op_algorithm = projection_algorithm
        self.metric = metric
        self.n_min_projections = n_min_projections
        self.n_max_projections = n_max_projections

        self.stats = None

    def benchmark(self, X, y):
        self.stats = list()
        if self.n_max_projections > X.shape[1]:
            # TODO: replace with normal logging system
            print(f'[WARNING] rank of projection matrix is higher than features matrix, setting n_max_projections to {X.shape[1]}')
            self.n_max_projections = X.shape[1]

        for j in range(self.n_min_projections, self.n_max_projections + 1):
            stats_per_j = list()
            self.op_algorithm.set_projections_num(j)
            for i in tqdm(range(self.n_holdout_validations)):
                X_pca = self.op_algorithm.fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.5, shuffle=True)

                estimator = deepcopy(self.estimator)
                estimator.fit(X_train, y_train)

                score = self.metric(estimator.predict(X_test), y_test)
                stats_per_j.append(score)
            self.stats.append(stats.describe(stats_per_j))

        return self.stats
