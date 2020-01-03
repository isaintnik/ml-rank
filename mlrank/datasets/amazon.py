import pandas as pd
import numpy as np

from mlrank.datasets.dataset import dataframe_to_series_map, fit_encoder, SeparatedDataset, HoldoutDataset, \
    get_features_except


class AmazonDataSet(SeparatedDataset):
    def __init__(self, train_folder: str, test_folder: str):
        super().__init__('amazon', train_folder, test_folder)
        self.encoders = None

        self.cat_features = ['ACTION', 'RESOURCE', 'MGR_ID', 'ROLE_ROLLUP_1',
       'ROLE_ROLLUP_2', 'ROLE_DEPTNAME', 'ROLE_TITLE', 'ROLE_FAMILY_DESC',
       'ROLE_FAMILY', 'ROLE_CODE']

        self.train_plain = None
        self.train_transformed = None
        self.test_plain = None
        self.test_transformed = None

    def get_continuous_feature_names(self):
        return []

    def load_train_from_file(self):
        self.train = pd.read_csv(self.train_folder)
        self.train = self.train.sample(frac=1)
        self.train.drop('Unnamed: 0', axis=1, inplace=True)
        self.train.dropna(inplace=True)

    def load_test_from_file(self):
        self.test = pd.read_csv(self.test_folder)
        self.test = self.test.sample(frac=1)
        self.test.drop('Unnamed: 0', axis=1, inplace=True)
        self.test.dropna(inplace=True)

    def process_features(self):
        encoders = dict()
        for feature in self.cat_features:
            #encoders[feature] = fit_encoder(self.data[feature])
            encoders[feature] = fit_encoder(pd.concat([self.train[feature], self.test[feature]]))

            #self.data[feature] = encoders[feature].transform(self.data[feature])
            self.train[feature] = np.squeeze(encoders[feature].transform(self.train[feature]))
            self.test[feature] = np.squeeze(encoders[feature].transform(self.test[feature]))

        self.features_ready = True

    def get_dummies(self, data_chunk: pd.DataFrame) -> dict:
        dummy_features = dict()

        for feature in self.cat_features:
            if feature != 'ACTION':
                dummy_features[feature] = pd.get_dummies(data_chunk[feature]).values.T

        return dummy_features

    def cache_features(self):
        self.train_plain = dataframe_to_series_map(get_features_except(self.train, ['ACTION']))
        self.train_transformed = self.get_dummies(get_features_except(self.train, ['ACTION']))

        self.test_plain = dataframe_to_series_map(get_features_except(self.test, ['ACTION']))
        self.test_transformed = self.get_dummies(get_features_except(self.test, ['ACTION']))

    def get_test_target(self) -> pd.Series:
        return self.test['ACTION'].values#.reshape(-1, 1)

    def get_train_target(self) -> pd.Series:
        return self.train['ACTION'].values#.reshape(-1, 1)

    def get_train_features(self, convert_to_linear: bool) -> dict:
        if not self.features_ready:
            raise Exception('call process_features')

        if not convert_to_linear:
            return self.train_plain
        else:
            return self.train_transformed

    def get_test_features(self, convert_to_linear: bool) -> dict:
        if not self.features_ready:
            raise Exception('call process_features')

        if not convert_to_linear:
            return self.test_plain
        else:
            return self.test_transformed
