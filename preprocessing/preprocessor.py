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

    def explain_emoticons(self, tokens):
        return [self.emoticons[w] if w in self.emoticons else w for w in tokens]

    def load_emoticons(self):
        with open(self.EMOTICONS_PATH) as emoticons:
            return json.load(emoticons)
