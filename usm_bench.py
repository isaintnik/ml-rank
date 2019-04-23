import warnings
import sys
import os

from sklearn.exceptions import ConvergenceWarning

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

from mlrank.benchmarks.submodular.usm import USMBenchmark
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.externals import joblib

from mlrank.synth.loader import DataLoader

if __name__ == '__main__':
    treshold = 0.999

    benchmark = USMBenchmark(LogisticRegression(solver='liblinear', multi_class='auto'), accuracy_score, treshold=treshold, n_cv=1000, train_share=.8)

    usm_features = joblib.load('/Users/ppogorelov/Python/github/ml-rank/data/mlrank_realdata.bin')

    X, y = DataLoader().load_data_lung_cancer('./datasets/lung-cancer.data')

    data = list()
    for i in usm_features['lung_cancer, MLPClassifier']:
       print(benchmark.evaluate(X, y, i['result']))#, np.sum(i['result'] >= treshold))

    print(benchmark.evaluate(X, y, [1] * X.shape[1]))

    #benchmark = USMBenchmark(LogisticRegression(solver='liblinear'), accuracy_score, treshold=treshold, n_cv=100, train_share=.8)
    #
    #usm_features = joblib.load('/Users/ppogorelov/Python/github/ml-rank/data/mlrank_stat_cancer.bin')
    #
    #X, y = load_data()
    #
    #for i in usm_features['LogisticRegression']:
    #    print(benchmark.evaluate(X, y, i['result']), np.sum(i['result'] >= treshold))
    #
    #print(benchmark.evaluate(X, y, [1] * X.shape[1]))