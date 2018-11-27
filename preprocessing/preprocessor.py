import ipdb
import html
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import pandas as pd
import json
# from regexp_manager import RegexpManager


class TextPreprocessor:
    # def __init__(self):
    #     regexp_manager = RegexpManager()
    #     self.regexps = regexp_manager.get_compiled()

    def preprocess(self, text):
        # summarized = self.summarize_contents(text)
        tokens = self.tokenize(text)
        tokens_without_stopwords = self.remove_stopwords(tokens)
        return tokens_without_stopwords

    def tokenize(self, sent):
        tokenizer = TweetTokenizer(strip_handles=True,
                                   preserve_case=True, reduce_len=True)
        return tokenizer.tokenize(sent)

    # def stemmize_tokens(self, tokens):
    #     stemmer = SnowballStemmer("english")
    #     return [stemmer.stem(token) for token in tokens]

    # def summarize_contents(self, text):
    #     ipdb.set_trace()
    #     for item, regexp in self.regexps.items():
    #         ipdb.set_trace()
    #         regexp.sub(lambda m: " " + "<" + item + ">" + " ",
    #                    text)
    #     return text

    def remove_stopwords(self, tokens):
        stop_words = set(stopwords.words('english'))
        return [token for token in tokens if not token in stop_words]


# TEST_SET = "data/train/original/twitter-2013test-A.tsv"
# test_data = pd.read_csv(TEST_SET, sep='\t',
#                         header=None, names=["id", "sentiment", "text"],
#                         usecols=["sentiment", "text"])

# pre = TextPreprocessor()

# ipdb.set_trace()
