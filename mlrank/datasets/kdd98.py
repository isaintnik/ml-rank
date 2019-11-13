import pandas as pd
import numpy as np

from mlrank.datasets.dataset import dataframe_to_series_map, fit_encoder, SeparatedDataset


class KDD99DataSet(SeparatedDataset):
    def __init__(self, train_folder: str, test_folder: str):
        super().__init__('kdd98', train_folder=train_folder, test_folder=test_folder)

        self.encoders = dict()