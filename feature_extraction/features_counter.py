import ipdb
from collections import Counter
from math import log2
import numpy as np
from feature_extraction.negation_marker import NegationMarker


class FeaturesCounter:
    def __init__(self):
        self.neg_marker = NegationMarker()

    def count_features(self, df):
        marked_docs = self.neg_marker.mark_docs_negations(df.text)
        sent_scores = self.count_sent_scores(df)

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

    def count_sent_scores(self, df):
        df = df.loc[lambda df: df.sentiment != "neutral"]

        sent_totals, freqs, lexicon = self.get_totals_freqs_and_lexicon(df)

        sent_scores = {}

        for token in lexicon:
            pos = freqs["positive"][token] * sent_totals["negative"]
            neg = freqs["negative"][token] * sent_totals["positive"]
            sent_scores[token] = log2(pos / neg)

        return sent_scores

    def get_totals_freqs_and_lexicon(self, df):
        sent_totals = Counter()

        freqs = {
            "positive": Counter(),
            "negative": Counter()
        }

        lexicon = []

        for doc, sentiment in zip(df.text, df.sentiment):
            clean = [token for token in doc.split(
            ) if not self.not_feature_token(token)]

            for token in clean:
                freqs[sentiment][token] += 1

        # Removing of tokens that appears
        # less than 5 times in each of groups
        for sent in list(freqs):
            for token in list(freqs[sent]):
                if not self.appears_less_than_in_each(token, freqs, 5):
                    freqs[sent].pop(token)
                else:
                    lexicon.append(token)
                    sent_totals[sent] += freqs[sent][token]

        return sent_totals, freqs, set(lexicon)

    def not_feature_token(self, token):
        return self.neg_marker.is_back_off_punct_or_num(token) and token != "but" and not self.neg_marker.negations.match(token)

    def appears_less_than_in_each(self, token, freqs, times=5):
        appears = True
        for sent in list(freqs):
            if freqs[sent][token] < times:
                appears = False
                break

        return appears
