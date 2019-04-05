import numpy as np
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

from mlrank.submodularity.functions.metrics_prediction import mutual_information_normalized
from mlrank.submodularity.optimization.ffs import ForwardFeatureSelection
from mlrank.synth.linear import LinearProblemGenerator
from mlrank.preprocessing.dichtomizer import MaxentropyMedianDichtomizationTransformer, dichtomize_matrix
#from mlrank.submodularity.functions.metrics_prediction import mutual_information
from mlrank.submodularity.functions.metrics_dataset import (
    joint_entropy_score_estimate,
    joint_entropy_score_exact,
    joint_entropy_score_ica_estimate,
    informational_regularization_2)

joint_entropy_score = joint_entropy_score_estimate

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def calc_loss(subset, X_d, X_c, y, decision_function, n_bins):
    return mutual_information_normalized(
               X_d[:, subset], y,
               decision_function, n_bins=n_bins
           ) + \
           informational_regularization_2(
               subset, X_d, X_c, decision_function, n_bins=n_bins
           )


if __name__ == '__main__':
    np.random.seed(42)

    y, ground, noise, corr = LinearProblemGenerator.make_correlated_uniform(500, 5, 7, 5)
    X = np.hstack([ground, noise, corr])

    for i in [4, 8]:
        X_dicht = dichtomize_matrix(X, i)

        print('-'*50)
        print('n_bin: ', i)
        print('-' * 50)

        ffs = ForwardFeatureSelection(n_bins=i, lambda_=1)
        subset_a = ffs.select(X_dicht, X, y, 5, LinearRegression(), extra_loss=False)
        print('ffs (no reg)', subset_a)
        print('loss (no reg)', calc_loss(subset_a, X_dicht, X, y, LinearRegression(), n_bins=i))
        print('loss (ground)', calc_loss(list(range(8)), X_dicht, X, y, LinearRegression(), n_bins=i))
        for lambda_ in [.1, .3, .6, .9]:
            print('#' * 50)
            print('lambda ', lambda_)
            ffs = ForwardFeatureSelection(n_bins=i, lambda_=lambda_)
            subset_b = ffs.select(X_dicht, X, y, 5, LinearRegression(), extra_loss=True)
            print('ffs (reg)', subset_b)
            print('loss (reg)', calc_loss(subset_b, X_dicht, X, y, LinearRegression(), n_bins=i))
