from feature_extraction.automatic_features_counter import AutomaticFeaturesCounter
from feature_extraction.manual_features_counter import ManualFeaturesCounter
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class FeatureExtractor:
    def __init__(self, data_loader=None, logger=None):
        self.logger = logger
        self.data_loader = data_loader
        self.scaler = MinMaxScaler()
        # self.basic_feats_ctr = BasicFeaturesCounter()
        self.afc = AutomaticFeaturesCounter()
        self.mfc = ManualFeaturesCounter()

    def get_train_test_features(self, manual=True, auto=True,
                                scaled=False, ternary=True):
        if auto or manual:
            f_train, f_test = self.data_loader.get_train_test(ternary=ternary)
            self.manual = manual
            self.auto = auto
            self.scaled = scaled
            return self.get_features(f_train), self.get_features(f_test)
        else:
            return None, None

    def get_features(self, df):
        manual_features = np.array([])
        auto_features = np.array([])

        if self.manual:
            manual_features = self.mfc.get_features(df)

        if self.auto:
            auto_features = self.afc.get_features(df)

        self.logger.write("manual features: %s" % self.manual)
        self.logger.write("auto_features: %s" % self.auto)
        self.logger.write("min-max scaling: %s" % self.scaled)

        features = np.concatenate((manual_features, auto_features), axis=1)

        if self.scaled:
            return self.scaler.fit_transform(features)
        else:
            return features
