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
    informational_regularization_2)

joint_entropy_score = joint_entropy_score_ica_estimate

if __name__ == '__main__':
    np.random.seed(42)
    y, ground, noise, corr = LinearProblemGenerator.make_correlated_uniform(100, 3, 5, 2)#(500, 10, 10, 5)

    X = np.hstack([ground, noise, corr])

    n_ground = ground.shape[1]
    n_noise = noise.shape[1]
    n_corr = corr.shape[1]

    decision_function = LinearRegression()

    _lambda = 1

    for i in [8, 16]:
        X_dicht = dichtomize_matrix(X, i)

        print(X_dicht.shape)

        a = list(range(n_ground, X.shape[1]))
        b = list(range(n_ground))
        c = list(range(0, X.shape[1]))
        d = np.random.choice(n_ground+n_noise+n_corr, 10).tolist()
        e = list(range(n_ground+n_noise, X.shape[1]))
        e = [0,1,2,-2]

        x_noise_info = mutual_information(a, X, y, decision_function, n_bins=i)
        x_dicht_info = mutual_information(b, X, y, decision_function, n_bins=i)
        x_full_info = mutual_information(c, X, y, decision_function, n_bins=i)
        x_rand_info = mutual_information(d, X, y, decision_function, n_bins=i)
        x_mc_info = mutual_information(e, X, y, decision_function, n_bins=i)
        x_pred_info = mutual_information(e, X, y, decision_function, n_bins=i)

        print('-' * 100)
        print('-' * 100)

        x_noise_reg = _lambda*informational_regularization_2(a, X_dicht, X, decision_function, n_bins=i)
        x_real_reg = _lambda*informational_regularization_2(b, X_dicht, X, decision_function, n_bins=i)
        x_full_reg = _lambda*informational_regularization_2(c, X_dicht, X, decision_function, n_bins=i)
        x_rand_reg = _lambda*informational_regularization_2(d, X_dicht, X, decision_function, n_bins=i)
        x_mc_reg = _lambda * informational_regularization_2(e, X_dicht, X, decision_function, n_bins=i)
        x_pred_reg = _lambda * informational_regularization_2(e, X_dicht, X, decision_function, n_bins=i)

        print(x_noise_info, x_noise_reg, ' delta: ', x_noise_info + x_noise_reg)
        print(x_dicht_info, x_real_reg, ' delta: ', x_dicht_info + x_real_reg)
        print(x_full_info, x_full_reg, ' delta: ', x_full_info + x_full_reg)
        print(x_rand_info, x_rand_reg, ' delta: ', x_rand_info + x_rand_reg)
        print(x_mc_info, x_mc_reg, ' delta: ', x_mc_info + x_mc_reg)

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