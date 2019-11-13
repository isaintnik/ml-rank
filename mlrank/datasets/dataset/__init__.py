import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .dataset import DataSet
from .holdout_dataset import HoldoutDataset
from .separated_dataset import SeparatedDataset


def dataframe_to_series_map(df: pd.DataFrame) -> dict:
    return {c: df[c].values.reshape(-1, 1) for c in df.columns}


def fit_encoder(classes) -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.fit(classes)
    return encoder

