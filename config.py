#from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier

from mlrank.datasets import BreastDataSet, AdultDataSet, AmazonDataSet
from mlrank.datasets.internet import InternetDataSet
from mlrank.datasets.seizures import SeizuresDataSet

BREAST_CANCER_PATH = './datasets/breast_cancer.csv'

AMAZON_TRAIN_PATH = './datasets/amazon_train.csv'
AMAZON_TEST_PATH = './datasets/amazon_train.csv'

ADULT_TRAIN_PATH = './datasets/adult_train.csv'
ADULT_TEST_PATH = './datasets/adult_test.csv'

INTERNET_TRAIN_PATH = './datasets/internet_train.dat'
INTERNET_TEST_PATH = './datasets/internet_test.dat'

SEIZURES_TRAIN_PATH = './datasets/seizures_train.csv'
SEIZURES_TEST_PATH = './datasets/seizures_test.csv'

CATBOOST_PARAMS = {
    'learning_rate': 0.03,
    'depth': 6,
    'fold_len_multiplier': 2,
    'rsm': 1.0,
    'border_count': 128,
    #'ctr_border_count': 16,
    'l2_leaf_reg': 3,
    'leaf_estimation_method': 'Newton',
    #'gradient_iterations': 10,
    'iterations': 10,
    #'ctr_description': ['Borders','CounterMax'],
    'used_ram_limit': 100000000000,
}


ALGO_PARAMS = {
    'dataset': [
        {'type': 'holdout', 'supported': ['linear', 'mlp', 'gbdt', 'cb'], 'problem': 'classification', 'name': "breast_cancer", 'data': BreastDataSet(BREAST_CANCER_PATH)},
        {'type': 'train_test', 'supported': ['linear', 'mlp', 'gbdt', 'cb'], 'problem': 'classification', 'name': "adult", 'data': AdultDataSet(ADULT_TRAIN_PATH, ADULT_TEST_PATH)},
        {'type': 'train_test', 'supported': ['linear', 'gbdt', 'cb'], 'problem': 'classification', 'name': "internet", 'data': InternetDataSet(INTERNET_TRAIN_PATH, INTERNET_TEST_PATH)},
        {'type': 'train_test', 'supported': ['linear', 'gbdt', 'cb'], 'problem': 'classification', 'name': "amazon", 'data': AmazonDataSet(AMAZON_TRAIN_PATH, AMAZON_TEST_PATH)},
        {'type': 'train_test', 'supported': ['linear', 'mlp', 'gbdt', 'cb'], 'problem': 'classification', 'name': "seizures", 'data': SeizuresDataSet(SEIZURES_TRAIN_PATH, SEIZURES_TEST_PATH)},
    ],

    'decision_function': [
        {'regression': Lasso(),
         'classification': LogisticRegression(
             multi_class='auto', solver='liblinear', penalty='l1', C=1000
         ), 'type': 'linear'},
        {'regression': LGBMRegressor(
            boosting_type='gbdt',
            learning_rate=0.05,
            num_iterations=1200,
            max_depth=5,
            n_estimators=1000,
            verbose=-1,
            num_leaves=2 ** 5,
            silent=True,
            n_jobs=4
            ),
        'classification': LGBMClassifier(
            boosting_type='gbdt',
            learning_rate=0.05,
            num_iterations=1200,
            max_depth=5,
            n_estimators=1000,
            verbose=-1,
            num_leaves=2 ** 5,
            silent=True,
            n_jobs=4
            ),
        'type': 'gbdt'
        },
        #{'classification': CatBoostClassifier(
        #    **{**CATBOOST_PARAMS, **{
        #        'loss_function': 'MultiClass',
        #        'verbose': False,
        #        'thread_count': 4,
        #        'random_seed': 0}
        #    }
        #), 'regression': CatBoostRegressor(
        #    **{**CATBOOST_PARAMS, **{
        #        'loss_function': 'RMSE',
        #        'verbose': False,
        #        'thread_count': 4,
        #        'random_seed': 0
        #    }}
        #), 'type': 'cb'},
        #{'regression': MLPRegressor(hidden_layer_sizes=(2, 2), activation='relu'),
        # 'classification': MLPClassifier(hidden_layer_sizes=(2, 2), activation='relu'),
        # 'type': 'mlp'
        #},
    ]
}

HYPERPARAMS = {
    #'bins': [2, 4, 8],
    'bins': [4],
    'lambda': [0.001]
}
