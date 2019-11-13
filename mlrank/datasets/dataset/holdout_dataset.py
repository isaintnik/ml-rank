import numpy as np
from mlrank.datasets.dataset import DataSet


class HoldoutDataset(DataSet):
    def __init__(self, name: str, data_folder: str):
        super().__init__(name)

        self.data_folder = data_folder

        self.data = None

    def get_features(self, convert_to_linear: bool) -> dict:
        raise NotImplementedError()

    def get_target(self) -> np.array:
        raise NotImplementedError()

