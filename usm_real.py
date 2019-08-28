import os
import sys
import warnings
from functools import partial

from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, mutual_info_score

from mlrank.preprocessing.dichtomizer import dichtomize_vector
from mlrank.submodular.metrics import mutual_information_regularized_score_penalized

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

import numpy as np
import pandas as pd

from sklearn import clone
from sklearn.externals import joblib
from sklearn.utils.multiclass import type_of_target

from itertools import product

from mlrank.submodular.optimization.usm import (
    MultilinearUSMExtended,
    MultilinearUSMClassic
)

from mlrank.benchmarks.holdout import HoldoutBenchmark, DichtomizedHoldoutBenchmark

# models
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.neural_network import MLPRegressor, MLPClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle


class DataLoader(object):
    @staticmethod
    def load_data_breast_cancer(path):
        # df = pd.read_csv('/Users/ppogorelov/Python/github/ml-rank/datasets/cancer/breast_cancer.csv')
        #df = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '/datasets/cancer/breast_cancer.csv')
        df = pd.read_csv(path)

        y = df.diagnosis.replace('M', 0).replace('B', 1).values
        X = np.asarray(df.drop(['diagnosis', 'id', 'Unnamed: 32'], axis=1).values)

        X, y = shuffle(X, y)

        return X, y.reshape(-1, 1)

    @staticmethod
    def load_data_heart_desease(path):
        df = pd.read_csv(
            path,
            sep=' ',
            names=['ag','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num',
        ]).dropna()

        y = df['num'].values
        X = df[list(set(df.columns).difference(['num']))].values

        X, y = shuffle(X, y)

        return X, y.reshape(-1, 1)

    @staticmethod
    def load_data_forest_fire(path):
        df = pd.read_csv(path).dropna()

        df['month'] = LabelEncoder().fit_transform(df['month'])
        df['day'] = LabelEncoder().fit_transform(df['day'])

        #y1 = np.log(df['area'] + 1)
        y = df['area'].values
        X = df[list(set(df.columns).difference(['area']))].values

        X, y = shuffle(X, y)

        return X, y.reshape(-1, 1)

    @staticmethod
    def load_data_forest_fire_log(path):
        df = pd.read_csv(path).dropna()

        df['month'] = LabelEncoder().fit_transform(df['month'])
        df['day'] = LabelEncoder().fit_transform(df['day'])

        y = np.log(df['area'].values + 1)
        X = df[list(set(df.columns).difference(['area']))].values

        X, y = shuffle(X, y)

        return X, y.reshape(-1, 1)

    @staticmethod
    def load_data_arrhythmia(path):
        df = pd.read_csv(path, header=None).replace('?', np.nan)

        valid_columns = df.columns[~(df.isna().sum(axis=0).sort_values(ascending=False) > 0).sort_index()]

        X = df.loc[:, valid_columns].loc[:, valid_columns[:-1]].values
        y = df.loc[:, valid_columns].loc[:, valid_columns[-1]].values

        print(np.sum(np.isnan(X)))

        X, y = shuffle(X, y)

        return X, y.reshape(-1, 1)

    @staticmethod
    def load_data_lung_cancer(path):
        df = pd.read_csv(path, header=None).replace('?', np.nan)

        valid_columns = df.columns[~(df.isna().sum(axis=0).sort_values(ascending=False) > 0).sort_index()]

        X = df.loc[:, valid_columns].loc[:, 1:].values
        y = df.loc[:, valid_columns].loc[:, 0].values

        X, y = shuffle(X, y)

        return X, y.reshape(-1, 1)

    @staticmethod
    def load_data_seizures(path):
        df = pd.read_csv(path)

        X = df[sorted(list(set(df.columns).difference(['y'])))].values
        y = df['y'].values

        # skipping column with string content
        X, y = shuffle(X[:, 1:], y)

        return X, y.reshape(-1, 1)


#BREAST_CANCER_PATH = '../datasets/breast_cancer.csv'
#ARRHYTHMIA_PATH = '../datasets/arrhythmia.data'
#FOREST_FIRE_PATH = '../datasets/forestfires.csv'
#HEART_DESEASE_PATH = '../datasets/reprocessed.hungarian.data'
#SEIZURES_PATH = '../datasets/seizures.csv'
#LUNG_CANCER_PATH = '../datasets/lung-cancer.data'

BREAST_CANCER_PATH = './datasets/breast_cancer.csv'
ARRHYTHMIA_PATH = './datasets/arrhythmia.data'
FOREST_FIRE_PATH = './datasets/forestfires.csv'
HEART_DESEASE_PATH = './datasets/reprocessed.hungarian.data'
SEIZURES_PATH = './datasets/seizures.csv'
LUNG_CANCER_PATH = './datasets/lung-cancer.data'

# algorithm params
ALGO_PARAMS = {
    'dataset': [
        {'problem': 'classification', 'name': "lung_cancer", 'data': DataLoader.load_data_lung_cancer(LUNG_CANCER_PATH)},
        #{'problem': 'classification', 'name': "forest_fire", 'data': DataLoader.load_data_forest_fire(FOREST_FIRE_PATH)},
        #{'problem': 'classification', 'name': "forest_fire_log", 'data': DataLoader.load_data_forest_fire_log(FOREST_FIRE_PATH)},
        #{'problem': 'classification', 'name': "arrhythmia", 'data': DataLoader.load_data_arrhythmia(ARRHYTHMIA_PATH)},
        {'problem': 'classification', 'name': "breast_cancer", 'data': DataLoader.load_data_breast_cancer(BREAST_CANCER_PATH)},
        {'problem': 'classification', 'name': "heart_desease", 'data': DataLoader.load_data_heart_desease(HEART_DESEASE_PATH)},
        #{'problem': 'classification', 'name': "seizures", 'data': DataLoader.load_data_seizures(SEIZURES_PATH)} # pretty heavy
    ],

    'decision_function': [
        {'regression': Lasso(),
         'classification': LogisticRegression(multi_class='auto', solver='liblinear', penalty='l2', C=1)},
        #{'regression': MLPRegressor(hidden_layer_sizes=(5, 5), activation='relu'),
        # 'classification': MLPClassifier(hidden_layer_sizes=(5, 5), activation='relu')},
        #{'regression': LGBMRegressor(
        #        boosting_type='rf',
        #        learning_rate=1e-2,
        #        max_depth=5,
        #        subsample=0.7,
        #        n_estimators=200,
        #        verbose=-1,
        #        subsample_freq=5,
        #        num_leaves=2**5,
        #        silent=True
        #    ),
        #'classification': LGBMClassifier(
        #        boosting_type='rf',
        #        learning_rate=1e-2,
        #        max_depth=5,
        #        subsample=0.7,
        #        n_estimators=200,
        #        verbose=-1,
        #        subsample_freq=5,
        #        num_leaves=2 ** 5,
        #        silent=True
        #    )
        #}
    ]
}

# hyperparameters
HYPERPARAMS = {
    'bins': [2, 4, 8],
    'lambda': [.0, .3, .6, 1.]
}


if __name__ == '__main__':
    np.random.seed(42)

    feature_selection_share=.5

    joblib.dump('test', "./data/testdoc.bin")

    results = {}

    for dataset, decision_function in product(ALGO_PARAMS['dataset'], ALGO_PARAMS['decision_function']):
        dfunc = decision_function[dataset['problem']]

        key = "{}, {}".format(dataset['name'], dfunc.__class__.__name__)

        print('>>', key)

        results[key] = list()

        X, y = dataset['data']

        prev_lambda_param = None

        for lambda_param, bins in product(HYPERPARAMS['lambda'], HYPERPARAMS['bins']):
            print(bins, lambda_param)
            if bins >= X.shape[0] * feature_selection_share:
                print(key, bins, 'very small dataset for such dichtomization.')
                continue

            if prev_lambda_param == lambda_param and type_of_target(y) != 'continuous':
                print('skipping dichtomization.')
                continue

            prev_lambda_param = lambda_param

            score_function = partial(mutual_information_regularized_score_penalized, _lambda=lambda_param, _gamma=0.1)

            #y = dichtomize_vector(y, n_bins=2, ordered=False)
            #dfunc.fit(X, y)
            #yy = dfunc.predict(X)
            #print(mutual_info_score(dfunc.predict(X), y))

            bench = DichtomizedHoldoutBenchmark(
                MultilinearUSMExtended(
                    decision_function=dfunc, score_function=score_function, n_bins=bins, me_eps=.15, n_jobs=1
                ),
                feature_selection_share=feature_selection_share,
                decision_function=dfunc,
                n_holdouts=100,
                n_bins=bins,
                n_jobs=8
            )

            predictions = bench.benchmark(X, y)

            results[key].append({
                'bins': bins,
                'lambda': lambda_param,
                'result': predictions
            })

            joblib.dump(results, "./data/mlrank_realdata_usm.bin")
