import pandas as pd
from mlrank.datasets.dataset import HoldoutDataset, dataframe_to_series_map


class BreastDataSet(HoldoutDataset):
    def __init__(self, path):
        super().__init__('breast', path)

        self.data = None

    def load_from_folder(self):
        self.data = pd.read_csv(self.data_folder)
        self.data = self.data.drop(['id', 'Unnamed: 32'], axis=1)

    def process_features(self):
        self.data.diagnosis = self.data.diagnosis.replace('M', 0).replace('B', 1)
        self.features_ready = True

    def get_features(self, convert_to_linear: bool):
        # dataset is linear anyway
        return dataframe_to_series_map(self.data[set(self.data.columns).difference('diagnosis')])

    def get_target(self):
        return self.data['diagnosis'].values
