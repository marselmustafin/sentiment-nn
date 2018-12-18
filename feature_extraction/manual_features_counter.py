import ipdb
from feature_extraction.negation_marker import NegationMarker
import numpy as np

class ManualFeaturesCounter:
    MPQA_LEXICON_PATH = "feature_extraction/lexicons/subjclueslen1-HLTEMNLP05.tff"

    def __init__(self):
        self.neg_marker = NegationMarker()

    def get_features(self, df):
        marked_docs = self.neg_marker.mark_docs_negations(df.text)
        sent_scores = self.get_sent_scores()

        features = []

        for doc in marked_docs:
            pos_aff = 0
            neg_aff = 0
            pos_neg = 0
            neg_neg = 0
            for token in doc:
                if token[-3:] == "NEG":
                    if sent_scores.get(token[:-4], 0) > 0:
                        pos_neg += sent_scores.get(token[:-4], 0)
                    else:
                        neg_neg += sent_scores.get(token[:-4], 0)
                else:
                    if sent_scores.get(token, 0) > 0:
                        pos_aff += sent_scores.get(token, 0)
                    else:
                        neg_aff += sent_scores.get(token, 0)
            features.append([pos_aff, neg_aff, pos_neg, neg_neg])

        return np.array([features])[0]


    def get_sent_scores(self):
        rows = self.read_data()
        sent_scores = {}

        for row in rows:
            if row['type'][:6] == "strong":
                score = 1
            elif row['type'][:4] == "weak":
                score = 2

            if row['priorpolarity'] == "negative":
                score *= -1

            sent_scores[row['word1']] = score

        return sent_scores

    def read_data(self):
        rows = []

        for line in open(self.MPQA_LEXICON_PATH, "r", encoding="utf-8").readlines():
            pairs = line.split()

            word_dict = {}

            for pair in pairs:
                key_value = pair.split("=")
                if len(key_value) is 2:
                    word_dict[key_value[0]] = key_value[1]

            rows.append(word_dict)

        return rows
