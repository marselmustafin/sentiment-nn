from preprocessing.preprocessor import TextPreprocessor
import pandas as pd
import ipdb
import csv
import html
from os import listdir


class DataLoader:
    TRAIN_FILES_PATHS = ["data/train/" + f for f in listdir("data/train")]
    TEST_FILES_PATHS = [
        "data/test/SemEval2017-task4-test.subtask-A.english.txt"]

    def __init__(self):
        self.train_data = self.read_data(self.TRAIN_FILES_PATHS)
        self.test_data = self.read_data(self.TEST_FILES_PATHS)
        self.preprocessor = TextPreprocessor()

    def get_train_test(self, ternary=False):
        train = self.train_data
        test = self.test_data

        if not ternary:
            train = self.train_data.loc[lambda df: df.sentiment != "neutral"]
            test = self.test_data.loc[lambda df: df.sentiment != "neutral"]

        train.text = self.preprocess_data(train.text)
        test.text = self.preprocess_data(test.text)

        return train, test

    def read_data(self, files):
        rows = []
        for f in files:
            rows += self.parse_file(f)
        return pd.DataFrame(data=rows, columns=["tweet_id", "sentiment", "text"])

    def preprocess_data(self, data):
        return data.apply(
            (lambda text: " ".join(self.preprocessor.preprocess(text))))

    def parse_file(self, file):
        rows = []
        for line in open(file, "r", encoding="utf-8").readlines():
            columns = line.rstrip().split("\t")
            tweet_id = columns[0]
            sentiment = columns[1]

            text = self.clean_text(" ".join(columns[2:]))

            if text != "Not Available":
                rows.append([tweet_id, sentiment, text])
        return rows

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


data_loader = DataLoader()
train, test = data_loader.get_train_test()
