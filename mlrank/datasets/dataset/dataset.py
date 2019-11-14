import pandas as pd
import joblib


class DataSet(object):
    def __init__(self, name):
        self.name = name
        self.data = None

        self.features_ready = False

    def process_features(self):
        raise NotImplementedError()

    # for dichtomization
    def get_continuous_feature_names(self):
        raise NotImplementedError()

    def cache_features(self):
        raise NotImplementedError()

    def get_name(self):
        return self.name