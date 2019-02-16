import numpy as np
import pandas as pd
from feature_extraction.negation_marker import NegationMarker
from itertools import tee
from sklearn.preprocessing import MinMaxScaler


class AutomaticFeaturesCounter:
    HS_LEXICON_UNIGRAM_SCORES_PATH = \
        "feature_extraction/lexicons/HS-AFFLEX-NEGLEX-unigrams.txt"
    HS_LEXICON_BIGRAM_SCORES_PATH = \
        "feature_extraction/lexicons/HS-AFFLEX-NEGLEX-bigrams.txt"
    LEXICON_140_UNIGRAM_SCORES_PATH = \
        "feature_extraction/lexicons/Emoticon-AFFLEX-NEGLEX-unigrams.txt"
    LEXICON_140_BIGRAM_SCORES_PATH = \
        "feature_extraction/lexicons/Emoticon-AFFLEX-NEGLEX-bigrams.txt"

    def __init__(self):
        self.neg_marker = NegationMarker()
        self.scaler = MinMaxScaler()

    def get_features(self, df):
        self.marked_docs = self.neg_marker.mark_docs_negations(df.text)

        uni_hs_scores = \
            self.get_uni_scores(self.HS_LEXICON_UNIGRAM_SCORES_PATH)
        uni_140_scores = \
            self.get_uni_scores(self.LEXICON_140_UNIGRAM_SCORES_PATH)

        uni_hs_features = self.count_uni_feats(uni_hs_scores)
        uni_140_features = self.count_uni_feats(uni_140_scores)

        big_hs_scores = \
            self.get_big_scores(self.HS_LEXICON_BIGRAM_SCORES_PATH)
        big_140_scores = \
            self.get_big_scores(self.LEXICON_140_BIGRAM_SCORES_PATH)

        big_hs_features = self.count_big_feats(big_hs_scores)
        big_140_features = self.count_big_feats(big_140_scores)

        features = np.concatenate((uni_hs_features, uni_140_features,
                                  big_hs_features, big_140_features), axis=1)

        scaled_features = self.scaler.fit_transform(features)

        return scaled_features

    def count_uni_feats(self, scores):
        features = []

        for doc in self.marked_docs:
            doc_scores = []

            for token in doc:
                score = scores.get(token, 0)

                if score == 0 and token[-8:] == "NEGFIRST":
                    score = scores.get(token[:-5], 0)

                doc_scores.append(score)

            features.append([np.count_nonzero(doc_scores), sum(doc_scores),
                            max(doc_scores), doc_scores[-1]])

        return np.array([features])[0]

    def count_big_feats(self, scores):
        features = []

        for doc in self.marked_docs:
            doc_scores = [scores.get((w1, w2), 0)
                          for w1, w2 in self.pairwise(doc)]

            # for preventing one word messages
            if doc_scores == []:
                doc_scores = [0]

            features.append([np.count_nonzero(doc_scores), sum(doc_scores),
                            max(doc_scores), doc_scores[-1]])

        return np.array([features])[0]

    def pairwise(self, iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    def get_uni_scores(self, path):
        data = pd.read_csv(path, sep="\t", header=None,
                           names=["word", "score", "pc", "nc"], quoting=3)

        scores = {}

        for word, score in zip(data.word, data.score):
            scores[word] = score

        return scores

    def get_big_scores(self, path):
        data = pd.read_csv(path, sep="\t", header=None,
                           names=["pair", "score", "pc", "nc"],
                           quoting=3)

        scores = {}

        for pair, score in zip(data.pair, data.score):
            scores[tuple(pair.split())] = score

        return scores
