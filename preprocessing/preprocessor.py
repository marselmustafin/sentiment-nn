import html
import json

import ipdb
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from tqdm import tqdm

from preprocessing.regexp_manager import RegexpManager

"""
    Preprocessing steps:
    1. Generalizing ("83" => "<number>", "https://google.com" => "<url>", etc.)
    2. Tweet tokenizing ("Suuuuuchhhh a nice day:)" => ["Suuuchhh", "a", "nice", "day", ":)"])
    3. Emoticons explaining (":)" => "<happy>")
    4. Stop words removing ("the", "is", "are", ...)
"""

class TextPreprocessor:
    EMOTICONS_PATH = "preprocessing/emoticons.json"
    GENERALIZE_TYPES = ["url", "email", "percent",
                        "money", "phone", "user", "time", "date", "number"]

    def __init__(self):
        regexp_manager = RegexpManager()
        self.regexps = regexp_manager.get_compiled(only=self.GENERALIZE_TYPES)
        self.emoticons = self.load_emoticons()

    def pre_process_doc(self, doc):
        generalized = self.generalize(doc)
        tokens = self.tokenize(generalized)
        with_emoticons = self.explain_emoticons(tokens)
        tokens_without_stopwords = self.remove_stopwords(with_emoticons)

        return tokens_without_stopwords

    def pre_process_docs(self, docs):
        for d in tqdm(docs, desc="PreProcessing..."):
            yield self.pre_process_doc(d)

    def tokenize(self, sent):
        tokenizer = TweetTokenizer(strip_handles=True,
                                   preserve_case=False,
                                   reduce_len=True)

        return tokenizer.tokenize(sent)

    def generalize(self, text):
        for item, regexp in self.regexps.items():
            text = regexp.sub(lambda m: " " + "<" + item + ">" + " ",
                              text)
        return text

<<<<<<< HEAD
    def summarize_contents(self, text):
        ipdb.set_trace()
        for item, regexp in self.regexps.items():
            ipdb.set_trace()
            regexp.sub(lambda m: " " + "<" + item + ">" + " ",
                       text)
        return text
=======
    def explain_emoticons(self, tokens):
        return [self.emoticons[w] if w in self.emoticons else w for w in tokens]
>>>>>>> 455405d... Add some improvements

    def remove_stopwords(self, tokens):
        stop_words = set(stopwords.words('english'))
        return [token for token in tokens if not token in stop_words]

<<<<<<< HEAD
    def clean_text(self, text):
        text = text.rstrip()

        if '""' in text:
            if text[0] == text[-1] == '"':
                text = text[1:-1]
            text = text.replace('\\""', '"')
            text = text.replace('""', '"')

        text = text.replace('\\""', '"')

        text = html.unescape(text)
        text = ' '.join(text.split())
        return text


# TEST_SET = "data/train/original/twitter-2013test-A.tsv"
# test_data = pd.read_csv(TEST_SET, sep='\t',
#                         header=None, names=["id", "sentiment", "text"],
#                         usecols=["sentiment", "text"])

pre = TextPreprocessor()

ipdb.set_trace()
=======
    def load_emoticons(self):
        with open(self.EMOTICONS_PATH) as emoticons:
            return json.load(emoticons)

pre = TextPreprocessor()
>>>>>>> 455405d... Add some improvements
