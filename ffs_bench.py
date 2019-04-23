import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from mlrank.benchmarks.submodular.ffs import FFSBenchmark
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from mlrank.synth.loader import DataLoader


def load_data():
    import os

    #df = pd.read_csv('/Users/ppogorelov/Python/github/ml-rank/datasets/cancer/breast_cancer.csv')
    df = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '/datasets/breast_cancer.csv')

    y = df.diagnosis.replace('M', 0).replace('B', 1).values
    X = np.asarray(df.drop(['diagnosis', 'id', 'Unnamed: 32'], axis=1).values)

    X = StandardScaler().fit_transform(X)

    return X, y.reshape(-1, 1)


if __name__ == '__main__':
    benchmark = FFSBenchmark(LogisticRegression(solver='liblinear', multi_class='auto', penalty='l1', C=2), accuracy_score, treshold=.95, n_cv=1000,
                             train_share=.8, n_cv_ffs=6)

    X, y = DataLoader().load_data_lung_cancer('./datasets/lung-cancer.data')
    #X, y = load_data()

    for i in range(1, X.shape[1]):
        print(benchmark.evaluate(X, y, i))