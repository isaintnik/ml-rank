import pandas as pd
import joblib


class DataSet(object):
    def __init__(self, name):
        self.name = name
        self.data = None

        self.features_ready = False

    def process_features(self):
        raise NotImplementedError()

    def get_name(self):
        return self.name