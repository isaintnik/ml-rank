import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from mlrank.synth.linear import LinearProblemGenerator
from mlrank.preprocessing.dichtomizer import dichtomize_matrix
from mlrank.submodularity.optimization.usm import MultilinearUSM

if __name__ == '__main__':
    np.random.seed(42)
    y, ground, noise, corr = LinearProblemGenerator.make_mc_uniform(40, 3, 5, 2)#(500, 10, 10, 5)

    X = np.hstack([ground, noise, corr])

    n_ground = ground.shape[1]
    n_noise = noise.shape[1]
    n_corr = corr.shape[1]

    decision_function = LinearRegression()

    for i in [4, 8, 16]:
        ums = MultilinearUSM(decision_function, i, .1)

        print(ums.select(X, y))
