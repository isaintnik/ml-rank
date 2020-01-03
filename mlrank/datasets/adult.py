import pandas as pd
import numpy as np
from mlrank.datasets.dataset import SeparatedDataset, dataframe_to_series_map, fit_encoder, get_features_except


class AdultDataSet(SeparatedDataset):
    def __init__(self, train_folder: str, test_folder: str):
        super().__init__('adult', train_folder, test_folder)
        self.encoders = None

        self.cat_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'income', 'race', 'sex']

        self.train_plain = None
        self.train_transformed = None
        self.test_plain = None
        self.test_transformed = None

    def load_train_from_file(self):
        self.train = pd.read_csv(self.train_folder)
        self.train = self.train.sample(frac=1)
        self.train.drop(['Unnamed: 0', 'native-country'], axis=1, inplace=True)
        self.train.dropna(inplace=True)

    def load_test_from_file(self):
        self.test = pd.read_csv(self.test_folder)
        self.test = self.test.sample(frac=1)
        self.test.drop(['Unnamed: 0', 'native-country'], axis=1, inplace=True)
        self.test.dropna(inplace=True)

    def process_features(self):
        self.train.drop('education', axis=1, inplace=True)
        self.test.drop('education', axis=1, inplace=True)

        #, 'native-country']
        self.encoders = dict()

        encoders = dict()
        for feature in self.cat_features:
            encoders[feature] = fit_encoder(pd.concat([self.train[feature], self.test[feature]]))

            self.train[feature] = np.squeeze(encoders[feature].transform(self.train[feature]))
            self.test[feature] = np.squeeze(encoders[feature].transform(self.test[feature]))

        self.features_ready = True

    def get_continuous_feature_names(self):
        return ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week', 'education-num']

    def get_dummies(self, data_chunk: pd.DataFrame) -> dict:
        dummy_workclass = pd.get_dummies(data_chunk['workclass'])
        #dummy_education_num = pd.get_dummies(data_chunk['education-num'])
        dummy_marital_status = pd.get_dummies(data_chunk['marital-status'])
        dummy_occupation = pd.get_dummies(data_chunk['occupation'])
        dummy_relationship = pd.get_dummies(data_chunk['relationship'])
        dummy_race = pd.get_dummies(data_chunk['race'])
        dummy_sex = pd.get_dummies(data_chunk['sex'])
        #dummy_native_country = pd.get_dummies(data_chunk['native-country'])

        return {
            'age': data_chunk['age'].values,
            'fnlwgt': data_chunk['fnlwgt'].values,
            'capital-gain': data_chunk['capital-gain'].values,
            'capital-loss': data_chunk['capital-loss'].values,
            'hours-per-week': data_chunk['hours-per-week'].values,
            'education-num': data_chunk['education-num'].values,
            'workclass': dummy_workclass.values.T,
            'marital-status': dummy_marital_status.values.T,
            'occupation': dummy_occupation.values.T,
            'relationship': dummy_relationship.values.T,
            'race': dummy_race.values.T,
            'sex': dummy_sex.values.T,
            #'native-country': dummy_native_country.values
        }

    def cache_features(self):
        self.train_plain = dataframe_to_series_map(get_features_except(self.train, ['income']))
        self.train_transformed = self.get_dummies(get_features_except(self.train, ['income']))

        self.test_plain = dataframe_to_series_map(get_features_except(self.test, ['income']))
        self.test_transformed = self.get_dummies(get_features_except(self.test, ['income']))

    def get_test_target(self) -> pd.Series:
        return self.test['income'].values#.reshape(-1, 1)

    def get_train_target(self) -> pd.Series:
        return self.train['income'].values#.reshape(-1, 1)

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