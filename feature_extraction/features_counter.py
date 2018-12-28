from collections import Counter
from math import log2
import numpy as np
import pandas as pd
from negation_marker import NegationMarker


class FeaturesCounter:
    def __init__(self):
        self.neg_marker = NegationMarker()

    # def count_features(self, df):
    #     marked_docs = self.neg_marker.mark_docs_negations(df.text)
    #     sent_scores = self.count_sent_scores(df)

    def save_lexicon_with_scores(self, df):
        sent_scores = self.count_sent_scores(df)
        result_data = {"word": list(
            sent_scores.keys()), "score": list(sent_scores.values())}

        result = pd.DataFrame(data=result_data)

        with open('neg_lex_scores.csv', 'a') as f:
            result.to_csv(f, header=False, index=False)

    def count_sent_scores(self, df):
        # df = df.loc[lambda df: df.sentiment != "neutral"]

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

        sent_map = {0: "negative", 4: "positive"}

        for doc, sentiment in zip(df.text, df.sentiment):
            clean = [token for token in doc.split(
            ) if not self.not_feature_token(token)]

            for token in clean:
                freqs[sent_map[sentiment]][token] += 1

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
        return self.neg_marker.is_back_off_punct_or_num(token) or token == "but" or self.neg_marker.negations.match(token)

    def appears_less_than_in_each(self, token, freqs, times=5):
        appears = True
        for sent in list(freqs):
            if freqs[sent][token] < times:
                appears = False
                break

        return appears

fc = FeaturesCounter()

data = pd.read_csv("feature_extraction/corpora/neg_140_corp.csv", header=None, names=[
                   "sentiment", "text"])

fc.save_lexicon_with_scores(data)
