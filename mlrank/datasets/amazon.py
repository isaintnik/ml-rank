import pandas as pd

from mlrank.datasets.dataset import dataframe_to_series_map, fit_encoder, SeparatedDataset, HoldoutDataset


class AmazonDataSet(HoldoutDataset):
    def __init__(self, folder: str):
        super().__init__('amazon', folder)

        self.cat_features = ['ACTION', 'RESOURCE', 'MGR_ID', 'ROLE_ROLLUP_1',
       'ROLE_ROLLUP_2', 'ROLE_DEPTNAME', 'ROLE_TITLE', 'ROLE_FAMILY_DESC',
       'ROLE_FAMILY', 'ROLE_CODE']
        
        self.encoders = dict()

    def load_from_folder(self):
        self.data = pd.read_csv(self.data_folder)
        self.data = self.data.sample(frac=1)
        self.data.drop('Unnamed: 0', axis=1, inplace=True)
        self.data.dropna(inplace=True)

    def process_features(self):
        encoders = dict()
        for feature in self.cat_features:
            encoders[feature] = fit_encoder(self.data[feature])

            self.data[feature] = encoders[feature].transform(self.data[feature])

        self.features_ready = True

    def get_dummies(self, data_chunk: pd.DataFrame) -> dict:
        dummy_features = dict()

        for feature in self.cat_features:
            if feature != 'ACTION':
                dummy_features[feature] = pd.get_dummies(data_chunk[feature]).values

        return dummy_features

    def get_target(self) -> pd.Series:
        return self.data['ACTION'].values.reshape(-1, 1)

    def get_features(self, convert_to_linear: bool) -> dict:
        if not self.features_ready:
            raise Exception('call process_features')

        if not convert_to_linear:
            return dataframe_to_series_map(self.data[set(self.data.columns).difference({'ACTION'})])
        else:
            return self.get_dummies(self.data[set(self.data.columns).difference({'ACTION'})])
