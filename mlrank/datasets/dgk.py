import re

import pandas as pd
import numpy as np

from mlrank.datasets.dataset import dataframe_to_series_map, fit_encoder, SeparatedDataset, get_features_except


class DontGetKickedDataSet(SeparatedDataset):
    def __init__(self, train_folder: str, test_folder: str):
        super().__init__('dontgetkicked', train_folder, test_folder)
        self.encoders = None

        self.target_feature = "IsBadBuy"

        self.cat_features = {0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 23, 24, 25, 26, 27, 29, 31, 32, 33, 34}

        self.train_plain = None
        self.train_transformed = None
        self.test_plain = None
        self.test_transformed = None

    def get_continuous_feature_names(self):
        return []

    @staticmethod
    def clean_string(s):
        return re.sub('[^A-Za-z0-9]+', "_", str(s))

    def prepare_dataset(self, data):
        data = data.copy()
        for i in self.cat_features:
            data[data.columns[i]] = data[data.columns[i]].apply(
                DontGetKickedDataSet.clean_string
            )

        columns_to_impute = []
        for i, column in enumerate(data.columns):
            if i not in self.cat_features and pd.isnull(data[column]).any():
                columns_to_impute.append(column)

        for column_name in columns_to_impute:
            data[column_name + "_imputed"] = pd.isnull(data[column_name]).astype(float)
            data[column_name].fillna(0, inplace=True)

        for i, column in enumerate(data.columns):
            if i not in self.cat_features:
                data[column] = data[column].astype(float)

        return data

    def load_train_from_file(self):
        self.train = pd.read_csv(self.train_folder)
        self.train = self.train.sample(frac=1)
        self.train = self.prepare_dataset(self.train)
        self.train.dropna(inplace=True)

    def load_test_from_file(self):
        self.test = pd.read_csv(self.test_folder)
        self.test = self.test.sample(frac=1)
        self.train = self.prepare_dataset(self.train)
        self.test.dropna(inplace=True)

    def process_features(self):
        encoders = dict()
        for i in self.cat_features:
            feature = self.train.columns[i]
            # encoders[feature] = fit_encoder(self.data[feature])
            encoders[feature] = fit_encoder(pd.concat([self.train[feature], self.test[feature]]))

            # self.data[feature] = encoders[feature].transform(self.data[feature])
            self.train[feature] = np.squeeze(encoders[feature].transform(self.train[feature]))
            self.test[feature] = np.squeeze(encoders[feature].transform(self.test[feature]))

        self.features_ready = True

    def get_dummies(self, data_chunk: pd.DataFrame) -> dict:
        dummy_features = dict()

        for feature in self.cat_features:
           dummy_features[feature] = pd.get_dummies(data_chunk[feature]).values.T

        return dummy_features

    def cache_features(self):
        self.train_plain = dataframe_to_series_map(get_features_except(self.train, [self.target_feature]))
        self.train_transformed = self.get_dummies(get_features_except(self.train, [self.target_feature]))

        self.test_plain = dataframe_to_series_map(get_features_except(self.test, [self.target_feature]))
        self.test_transformed = self.get_dummies(get_features_except(self.test, [self.target_feature]))

    def get_test_target(self) -> pd.Series:
        return self.test[self.target_feature].values  # .reshape(-1, 1)

    def get_train_target(self) -> pd.Series:
        return self.train[self.target_feature].values  # .reshape(-1, 1)

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
