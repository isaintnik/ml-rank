import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .dataset import DataSet
from .holdout_dataset import HoldoutDataset
from .separated_dataset import SeparatedDataset


def dataframe_to_series_map(df: pd.DataFrame) -> dict:
    return {c: df[c].values for c in df.columns}


def fit_encoder(classes) -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.fit(classes)
    return encoder


def get_features_except(features, _except):
    return features[list(set(features.columns).difference(set(_except)))]

