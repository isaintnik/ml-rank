import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from mlrank.synth.linear import LinearProblemGenerator
from mlrank.preprocessing.dichtomizer import dichtomize_matrix
from mlrank.submodularity.optimization.usm import MultilinearUSM

if __name__ == '__main__':
    np.random.seed(42)
    data = LinearProblemGenerator.make_mc_uniform(100, np.array([.1, 5, -3]), 2, 5)#(500, 10, 10, 5)

    X = np.hstack(data['features'])
    y = data['target']

    decision_function = LinearRegression()

    for i in [4, 8, 16]:
        ums = MultilinearUSM(decision_function, i, .3)

        print(ums.select(X, y))
