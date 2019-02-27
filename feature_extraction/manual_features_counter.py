from feature_extraction.negation_marker import NegationMarker
import numpy as np


class ManualFeaturesCounter:
    MPQA_LEXICON_PATH = \
        "feature_extraction/lexicons/subjclueslen1-HLTEMNLP05.tff"
    NRC_LEXICON_PATH = \
        "feature_extraction/lexicons/NRC_emotion_lexicon_list.txt"
    BING_LIU_POS_LEXICON_PATH = \
        "feature_extraction/lexicons/positive-words.txt"
    BING_LIU_NEG_LEXICON_PATH = \
        "feature_extraction/lexicons/negative-words.txt"

    def __init__(self):
        self.neg_marker = NegationMarker()
        self.scores = {"positive": 1, "negative": -1}

    def get_features(self, df):
        self.marked_docs = self.neg_marker.mark_docs_negations(df.text)

        mpqa_scores = self.get_mpqa_sent_scores()
        nrc_scores = self.get_nrc_sent_scores()
        bing_liu_scores = self.get_bing_liu_sent_scores()

        mpqa_feats = self.count_features(mpqa_scores)
        nrc_feats = self.count_features(nrc_scores)
        bing_liu_feats = self.count_features(bing_liu_scores)

        return np.concatenate((mpqa_feats, nrc_feats, bing_liu_feats), axis=1)

    def count_features(self, scores):

        features = []

        for doc in self.marked_docs:
            pos_aff = 0
            neg_aff = 0
            pos_neg = 0
            neg_neg = 0
            for token in doc:
                if token[-3:] == "NEG":
                    if scores.get(token[:-4], 0) > 0:
                        pos_neg += scores.get(token[:-4], 0)
                    else:
                        neg_neg += scores.get(token[:-4], 0)
                else:
                    if scores.get(token, 0) > 0:
                        pos_aff += scores.get(token, 0)
                    else:
                        neg_aff += scores.get(token, 0)
            features.append([pos_aff, neg_aff, pos_neg, neg_neg])

        return np.array([features])[0]

    def get_mpqa_sent_scores(self):
        rows = self.read_mpqa_data()
        sent_scores = {}

        for row in rows:
            if row['type'][:6] == "strong":
                score = 2
            elif row['type'][:4] == "weak":
                score = 1

            if row['priorpolarity'] == "negative":
                score *= -1

            sent_scores[row['word1']] = score

        return sent_scores

    def read_mpqa_data(self):
        rows = []

        for line in open(self.MPQA_LEXICON_PATH, "r").readlines():
            pairs = line.split()

            word_dict = {}

            for pair in pairs:
                key_value = pair.split("=")
                if len(key_value) is 2:
                    word_dict[key_value[0]] = key_value[1]

            rows.append(word_dict)

        return rows

    def get_nrc_sent_scores(self):
        scores = {}

        for line in open(self.NRC_LEXICON_PATH, "r").readlines():
            triplet = line.split()
            if triplet[1] == "positive" and triplet[2] == "1":
                scores[triplet[0]] = 1
            elif triplet[1] == "negative" and triplet[2] == "1":
                scores[triplet[0]] = -1

        return scores

    def get_bing_liu_sent_scores(self):
        sent_scores = {}

        for line in open(self.BING_LIU_POS_LEXICON_PATH, "r").readlines():
            sent_scores[line.strip()] = 1

        for line in open(self.BING_LIU_NEG_LEXICON_PATH, "r").readlines():
            sent_scores[line.strip()] = -1

        return sent_scores
