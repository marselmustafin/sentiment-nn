from feature_extraction.automatic_features_counter import AutomaticFeaturesCounter
from feature_extraction.manual_features_counter import ManualFeaturesCounter
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class FeatureExtractor:
    def __init__(self, logger=None):
        self.logger = logger
        self.scaler = MinMaxScaler()
        # self.basic_feats_ctr = BasicFeaturesCounter()
        self.afc = AutomaticFeaturesCounter()
        self.mfc = ManualFeaturesCounter()

    def get_features(self, df):
        manual_features = self.mfc.get_features(df)
        auto_features = self.afc.get_features(df)

        self.logger.write(
            "manual features: %s" % (False if manual_features is None else True))
        self.logger.write(
           "auto_features: %s" % (False if auto_features is None else True))

        features = np.concatenate((manual_features, auto_features), axis=1)
        scaled_features = self.scaler.fit_transform(features)

        return scaled_features
