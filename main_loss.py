import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from mlrank.synth.linear import LinearProblemGenerator
from mlrank.preprocessing.dichtomizer import MaxentropyMedianDichtomizationTransformer, dichtomize_matrix
from mlrank.submodularity.functions.metrics_prediction import mutual_information
from mlrank.submodularity.functions.metrics_dataset import (
    joint_entropy_score_estimate,
    joint_entropy_score_exact,
    joint_entropy_score_ica_estimate,
    informational_regularization_regression)

joint_entropy_score = joint_entropy_score_ica_estimate

if __name__ == '__main__':
    np.random.seed(42)
    y, ground, noise, corr = LinearProblemGenerator.make_mc_uniform(1000, 3, 5, 2)

    X = np.hstack([ground, noise, corr])

    n_ground = ground.shape[1]
    n_noise = noise.shape[1]
    n_corr = corr.shape[1]

    decision_function = LinearRegression()

    _lambda = 1

    for i in [4, 8, 16]:
        a = list(range(n_ground, X.shape[1]))
        b = list(range(n_ground))
        c = list(range(0, X.shape[1]))
        d = np.random.choice(n_ground+n_noise+n_corr, 10).tolist()
        e = list(range(n_ground+n_noise, X.shape[1]))
        f = [0,1,2,-1,-2]

        x_noise_info = mutual_information(a, X, y, decision_function, n_bins=i)
        x_real_info = mutual_information(b, X, y, decision_function, n_bins=i)
        x_full_info = mutual_information(c, X, y, decision_function, n_bins=i)
        x_rand_info = mutual_information(d, X, y, decision_function, n_bins=i)
        x_mc_info = mutual_information(e, X, y, decision_function, n_bins=i)
        x_pred_info = mutual_information(f, X, y, decision_function, n_bins=i)

        print('-' * 100)
        print('-' * 100)

        x_noise_reg = _lambda * informational_regularization_regression(a, X, decision_function, n_bins=i)
        x_real_reg = _lambda * informational_regularization_regression(b, X, decision_function, n_bins=i)
        x_full_reg = _lambda * informational_regularization_regression(c, X, decision_function, n_bins=i)
        x_rand_reg = _lambda * informational_regularization_regression(d, X, decision_function, n_bins=i)
        x_mc_reg = _lambda * informational_regularization_regression(e, X, decision_function, n_bins=i)
        x_pred_reg = _lambda * informational_regularization_regression(f, X, decision_function, n_bins=i)

        print('noise', x_noise_info, x_noise_reg, ' delta: ', x_noise_info + x_noise_reg)
        print('ground', x_real_info, x_real_reg, ' delta: ', x_real_info + x_real_reg)
        print('full', x_full_info, x_full_reg, ' delta: ', x_full_info + x_full_reg)
        print('random', x_rand_info, x_rand_reg, ' delta: ', x_rand_info + x_rand_reg)
        print('mc', x_mc_info, x_mc_reg, ' delta: ', x_mc_info + x_mc_reg)
        print('predicted', x_pred_info, x_pred_reg, ' delta: ', x_pred_info + x_pred_reg)

        #print('mutual information x dicht: ', x_dicht_info)
        #print('mutual information x raw: ', x_raw_info)
        #print('mutual information loss: ', x_raw_info - x_dicht_info)
        #
        #print('mutual information in ground set | whole set, ', joint_entropy_score([1,2,3,4,5], X_dicht))
        #print('mutual information in noise set | whole set ', joint_entropy_score(list(range(5, X_dicht.shape[1])), X_dicht))
        #print('mutual information in whole set | whole set ', joint_entropy_score(list(range(X_dicht.shape[1])), X_dicht))
        #
        #dataset_entropy = joint_entropy_score(list(range(X_dicht.shape[1])), X_dicht)
        #
        #print('submodular loss (ground): ', x_dicht_info - joint_entropy_score([1,2,3,4,5], X_dicht) / dataset_entropy)
        #print('submodular loss (noise): ', x_noise_info - joint_entropy_score(list(range(5, X_dicht.shape[1])), X_dicht) / dataset_entropy)