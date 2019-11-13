import numpy as np
from mlrank.datasets.dataset import DataSet


class SeparatedDataset(DataSet):
    def __init__(self, name: str, train_folder: str, test_folder: str):
        super().__init__(name)

        self.train_folder = train_folder
        self.test_folder = test_folder

        self.train = None
        self.test = None

    def get_train_features(self, convert_to_linear: bool) -> dict:
        raise NotImplementedError()

    def get_test_features(self, convert_to_linear: bool) -> dict:
        raise NotImplementedError()

    def get_train_target(self) -> np.array:
        raise NotImplementedError()

    def get_test_target(self) -> np.array:
        raise NotImplementedError()
