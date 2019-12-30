from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier

from mlrank.datasets import BreastDataSet, AdultDataSet, AmazonDataSet
from mlrank.datasets.internet import InternetDataSet

BREAST_CANCER_PATH = './datasets/breast_cancer.csv'
AMAZON_PATH = './datasets/amazon_train.csv'

ADULT_TRAIN_PATH = './datasets/adult_train.csv'
ADULT_TEST_PATH = './datasets/adult_test.csv'

INTERNET_TRAIN_PATH = './datasets/internet_train.dat'
INTERNET_TEST_PATH = './datasets/internet_test.dat'


ALGO_PARAMS = {
    'dataset': [
        {'type': 'holdout', 'problem': 'classification', 'name': "breast_cancer", 'data': BreastDataSet(BREAST_CANCER_PATH)},
        {'type': 'train_test', 'problem': 'classification', 'name': "adult", 'data': AdultDataSet(ADULT_TRAIN_PATH, ADULT_TEST_PATH)},
        {'type': 'train_test', 'problem': 'classification', 'name': "internet", 'data': InternetDataSet(INTERNET_TRAIN_PATH, INTERNET_TEST_PATH)},
        {'type': 'holdout', 'problem': 'classification', 'name': "amazon", 'data': AmazonDataSet(AMAZON_PATH)},
    ],

    'decision_function': [
        {'regression': Lasso(),
         'classification': LogisticRegression(
             multi_class='auto', solver='liblinear', penalty='l1', C=1000, n_jobs=6
         ), 'type': 'linear'},
        {'regression': MLPRegressor(hidden_layer_sizes=(3, 3), activation='relu'),
         'classification': MLPClassifier(hidden_layer_sizes=(3, 3), activation='relu'),
         'type': 'mlp'},
        {'regression': LGBMRegressor(
            boosting_type='gbdt',
            learning_rate=0.05,
            num_iterations=1200,
            max_depth=5,
            n_estimators=1000,
            verbose=-1,
            num_leaves=2 ** 5,
            silent=True
            ),
        'classification': LGBMClassifier(
            boosting_type='gbdt',
            learning_rate=0.05,
            num_iterations=1200,
            max_depth=5,
            n_estimators=1000,
            verbose=-1,
            num_leaves=2 ** 5,
            silent=True
            ),
        'type': 'gbdt'
        }
    ]
}

HYPERPARAMS = {
    'bins': [2, 4, 8],
    'lambda': [.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
}
