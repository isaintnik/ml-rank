import warnings


import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from mlrank.benchmarks.usm.benchmark import USMBenchmark
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.externals import joblib

def load_data():
    import os

    #df = pd.read_csv('/Users/ppogorelov/Python/github/ml-rank/datasets/cancer/breast_cancer.csv')
    df = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '/datasets/cancer/breast_cancer.csv')

    y = df.diagnosis.replace('M', 0).replace('B', 1).values
    X = np.asarray(df.drop(['diagnosis', 'id', 'Unnamed: 32'], axis=1).values)

    X = StandardScaler().fit_transform(X)

    return X, y.reshape(-1, 1)

if __name__ == '__main__':
    treshold = 0.5

    benchmark = USMBenchmark(LogisticRegression(solver='liblinear'), accuracy_score, treshold=treshold, n_cv=100, train_share=.8)

    usm_features = joblib.load('/Users/ppogorelov/Python/github/ml-rank/data/mlrank_stat_cancer.bin')

    X, y = load_data()

    for i in usm_features['LogisticRegression']:
        print(benchmark.evaluate(X, y, i['result']), np.sum(i['result'] >= treshold))

    print(benchmark.evaluate(X, y, [1] * X.shape[1]))