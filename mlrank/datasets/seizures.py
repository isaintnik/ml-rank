import re

import pandas as pd
import numpy as np

from mlrank.datasets.dataset import dataframe_to_series_map, fit_encoder, SeparatedDataset, get_features_except


class SeizuresDataSet(SeparatedDataset):
    def __init__(self, train_folder: str, test_folder: str):
        super().__init__('seizures', train_folder, test_folder)
        self.encoders = None

        self.target_feature = "y"

        self.train_plain = None
        self.train_transformed = None
        self.test_plain = None
        self.test_transformed = None

    def get_continuous_feature_names(self):
        return list(set(self.train_plain.keys()).difference({self.target_feature}))

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
        # nothing is required to do since there is no cat features
        self.features_ready = True

    def cache_features(self):
        self.train_transformed = self.train_plain = \
            dataframe_to_series_map(get_features_except(self.train, [self.target_feature]))

        self.test_transformed = self.test_plain = \
            dataframe_to_series_map(get_features_except(self.test, [self.target_feature]))

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
