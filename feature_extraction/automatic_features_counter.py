from collections import Counter
from math import log2
import numpy as np
import pandas as pd
from feature_extraction.negation_marker import NegationMarker


class AutomaticFeaturesCounter:
    def __init__(self):
        self.neg_marker = NegationMarker()

    def get_features(self, df):
        self.marked_docs = self.neg_marker.mark_docs_negations(df.text)

        aff_scores = self.get_aff_scores()
        neg_scores = self.get_neg_scores()

        return self.count_features(aff_scores, neg_scores)

    def count_features(self, aff_scores, neg_scores):
        features = []

        for doc in self.marked_docs:
            not_zero_count = 0
            overall_score = 0
            scores = []
            for token in doc:
                if token[-3:] == "NEG":
                    score = neg_scores.get(token[:-4], 0)
                else:
                    score = aff_scores.get(token, 0)
                scores.append(score)

                if score != 0:
                    not_zero_count += 1
                    overall_score += score

            features.append([not_zero_count, overall_score, max(scores), scores[-1]])

        return np.array([features])[0]

    def get_aff_scores(self):
        data = pd.read_csv("feature_extraction/lexicons/aff_140_lex_scores.csv", header=None, names=[
                           "word", "score"])
        scores = {}
        for word, score in zip(data.word, data.score):
            scores[word] = score

        return scores

    def get_neg_scores(self):
        data = pd.read_csv("feature_extraction/lexicons/neg_140_lex_scores.csv", header=None, names=[
                           "word", "score"])
        scores = {}
        for word, score in zip(data.word, data.score):
            scores[word] = score

        return scores
