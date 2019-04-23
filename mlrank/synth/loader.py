import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
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
        y = df['area']
        X = df[list(set(df.columns).difference(['area']))].values

        X, y = shuffle(X, y)

        return X, y.reshape(-1, 1)

    @staticmethod
    def load_data_forest_fire_log(path):
        df = pd.read_csv(path).dropna()

        df['month'] = LabelEncoder().fit_transform(df['month'])
        df['day'] = LabelEncoder().fit_transform(df['day'])

        y = np.log(df['area'] + 1)
        X = df[list(set(df.columns).difference(['area']))].values

        X, y = shuffle(X, y)

        return X, y.reshape(-1, 1)

    @staticmethod
    def load_data_arrhythmia(path):
        df = pd.read_csv(path, header=None).replace('?', np.nan)

        valid_columns = df.columns[~(df.isna().sum(axis=0).sort_values(ascending=False) > 0).sort_index()]

        X = df.loc[:, valid_columns].loc[:, df.columns[:-1]].values
        y = df.loc[:, valid_columns].loc[:, df.columns[-1]].values

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

        X = df[list(set(df.columns).difference(['y']))]
        y = df['y']

        X, y = shuffle(X, y)

        return X, y.reshape(-1, 1)
