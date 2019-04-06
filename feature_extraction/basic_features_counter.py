import re

class BasicFeaturesCounter:
    HASHTAG_REXEP = r"#(\w+)"

    def get_features(self, df):
        re.findall(self.HASHTAG_REXEP, s)
