import pandas as pd
from mlrank.datasets.dataset import HoldoutDataset, dataframe_to_series_map, get_features_except


class BreastDataSet(HoldoutDataset):
    def __init__(self, path):
        super().__init__('breast', path)

        self.train_plain = None
        self.train_transformed = None

        self.data = None

    def load_from_folder(self):
        self.data = pd.read_csv(self.data_folder)
        self.data = self.data.drop(['id', 'Unnamed: 32'], axis=1)

    def get_continuous_feature_names(self):
        return [
            "radius_mean",
            "texture_mean",
            "perimeter_mean",
            "area_mean",
            "smoothness_mean",
            "compactness_mean",
            "concavity_mean",
            "concave points_mean",
            "symmetry_mean",
            "fractal_dimension_mean",
            "radius_se",
            "texture_se",
            "perimeter_se",
            "area_se",
            "smoothness_se",
            "compactness_se",
            "concavity_se",
            "concave points_se",
            "symmetry_se",
            "fractal_dimension_se",
            "radius_worst",
            "texture_worst",
            "perimeter_worst",
            "area_worst",
            "smoothness_worst",
            "compactness_worst",
            "concavity_worst",
            "concave points_worst",
            "symmetry_worst",
            "fractal_dimension_worst"
            ]

    def process_features(self):
        self.data.diagnosis = self.data.diagnosis.replace('M', 0).replace('B', 1)
        self.features_ready = True

    def cache_features(self):
        self.train_plain = dataframe_to_series_map(get_features_except(self.data, ['diagnosis']))

    def get_features(self, convert_to_linear: bool):
        # dataset is linear anyway
        return self.train_plain

    def get_target(self):
        return self.data['diagnosis'].values
