import pandas as pd

from mlrank.datasets.dataset import dataframe_to_series_map, fit_encoder, SeparatedDataset


class AmazonDataSet(SeparatedDataset):
    def __init__(self, train_folder: str, test_folder: str):
        super().__init__('amazon', train_folder, test_folder)

        self.cat_features = ['ACTION', 'RESOURCE', 'MGR_ID', 'ROLE_ROLLUP_1',
       'ROLE_ROLLUP_2', 'ROLE_DEPTNAME', 'ROLE_TITLE', 'ROLE_FAMILY_DESC',
       'ROLE_FAMILY', 'ROLE_CODE']
        
        self.encoders = dict()

    def load_from_folder(self):
       pass

    def load_train_from_file(self):
        self.train = pd.read_csv(self.train_folder)
        self.train = self.train.sample(frac=1)
        self.train = self.train.drop('Unnamed: 0', axis=1)
        self.train = self.train.dropna()

    def load_test_from_file(self):
        self.test = pd.read_csv(self.test_folder)
        self.test = self.test.sample(frac=1)
        self.test = self.test.drop('Unnamed: 0', axis=1)
        self.test = self.test.dropna()

    def process_features(self):
        encoders = dict()
        for feature in self.cat_features:
            encoders[feature] = fit_encoder(pd.concat([self.train[feature], self.test[feature]]))

            self.train[feature] = encoders[feature].transform(self.train[feature])
            self.test[feature] = encoders[feature].transform(self.test[feature])

        self.features_ready = True

    def get_dummies(self, data_chunk: pd.DataFrame) -> dict:
        dummy_features = dict()

        for feature in self.cat_features:
            if feature != 'ACTION':
                dummy_features[feature] = pd.get_dummies(data_chunk[feature])

        return dummy_features

    def get_test_target(self) -> pd.Series:
        return self.test['ACTION'].values.reshape(-1, 1)

    def get_train_target(self) -> pd.Series:
        return self.train['ACTION'].values.reshape(-1, 1)

    def get_train_features(self, convert_to_linear: bool) -> dict:
        if not self.features_ready:
            raise Exception('call process_features')

        if not convert_to_linear:
            return dataframe_to_series_map(self.train[set(self.train.columns).difference({'ACTION'})])
        else:
            return self.get_dummies(self.train)

    def get_test_features(self, convert_to_linear: bool) -> dict:
        if not self.features_ready:
            raise Exception('call process_features')

        if not convert_to_linear:
            return dataframe_to_series_map(self.test[set(self.test.columns).difference({'ACTION'})])
        else:
            return self.get_dummies(self.test)