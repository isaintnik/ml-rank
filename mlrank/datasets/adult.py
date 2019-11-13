import pandas as pd
from mlrank.datasets.dataset import SeparatedDataset, dataframe_to_series_map, fit_encoder


class AdultDataSet(SeparatedDataset):
    def __init__(self, train_folder: str, test_folder: str):
        super().__init__('adult', train_folder, test_folder)
        self.encoders = None

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
        self.train.drop('education', axis=1, inplace=True)
        self.test.drop('education', axis=1, inplace=True)

        cat_features = ['workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'native-country', 'income', 'race', 'sex']
        self.encoders = dict()

        encoders = dict()
        for feature in cat_features:
            encoders[feature] = fit_encoder(pd.concat([self.train[feature], self.test[feature]]))

            self.train[feature] = encoders[feature].transform(self.train[feature])
            self.test[feature] = encoders[feature].transform(self.test[feature])

        self.features_ready = True

    def get_dummies(self, data_chunk: pd.DataFrame) -> dict:
        dummy_workclass = pd.get_dummies(data_chunk['workclass'])
        dummy_education_num = pd.get_dummies(data_chunk['education-num'])
        dummy_marital_status = pd.get_dummies(data_chunk['marital-status'])
        dummy_occupation = pd.get_dummies(data_chunk['occupation'])
        dummy_relationship = pd.get_dummies(data_chunk['relationship'])
        dummy_race = pd.get_dummies(data_chunk['race'])
        dummy_sex = pd.get_dummies(data_chunk['sex'])
        dummy_native_country = pd.get_dummies(data_chunk['native-country'])

        return {
            'age': data_chunk['age'].values.reshape(-1, 1),
            'fnlwgt': data_chunk['fnlwgt'].values.reshape(-1, 1),
            'capital-gain': data_chunk['capital-gain'].values.reshape(-1, 1),
            'capital-loss': data_chunk['capital-loss'].values.reshape(-1, 1),
            'hours-per-week': data_chunk['hours-per-week'].values.reshape(-1, 1),
            'workclass': dummy_workclass.values,
            'education-num': dummy_education_num.values,
            'marital-status': dummy_marital_status.values,
            'occupation': dummy_occupation.values,
            'relationship': dummy_relationship.values,
            'race': dummy_race.values,
            'sex': dummy_sex.values,
            'native-country': dummy_native_country.values
        }

    def get_test_target(self) -> pd.Series:
        return self.test['income'].values

    def get_train_target(self) -> pd.Series:
        return self.train['income'].values

    def get_train_features(self, convert_to_linear: bool) -> dict:
        if not self.features_ready:
            raise Exception('call process_features')

        if not convert_to_linear:
            return dataframe_to_series_map(self.train[set(self.train.columns).difference({'income'})])
        else:
            return self.get_dummies(self.train[set(self.train.columns).difference({'income'})])

    def get_test_features(self, convert_to_linear: bool) -> dict:
        if not self.features_ready:
            raise Exception('call process_features')

        if not convert_to_linear:
            return dataframe_to_series_map(self.test[set(self.test.columns).difference({'income'})])
        else:
            return self.get_dummies(self.test[set(self.test.columns).difference({'income'})])
