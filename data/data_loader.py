import pandas as pd
import html
from os import listdir


class DataLoader:
    TRAIN_FILES_PATHS = ["data/train/" + f for f in listdir("data/train")]
    TEST_FILES_PATHS = [
        "data/test/SemEval2017-task4-test.subtask-A.english.txt"]

    def __init__(self, preprocessor=None):
        self.preprocessor = preprocessor

    def get_train_test(self, ternary=False, randomize_train=False):
        return self.get_train(ternary=ternary, randomize=randomize_train), \
               self.get_test(ternary=ternary)

    def get_train(self, paths=None, ternary=False, randomize=False):
        train_data = self.read_data(paths or self.TRAIN_FILES_PATHS)
        train_set = self.get_set(train_data, ternary=ternary)

        return train_set.sample(frac=1) if randomize else train_set

    def get_test(self, paths=None, ternary=False):
        test_data = self.read_data(paths or self.TEST_FILES_PATHS)

        return self.get_set(test_data, ternary=ternary)

    def get_set(self, data, ternary=False):
        if not ternary:
            data = data[data.sentiment != "neutral"]

        if self.preprocessor:
            data.text = self.preprocess_data(data.text)

        return data

    def read_data(self, files):
        rows = []
        for f in files:
            rows += self.parse_file(f)
        return pd.DataFrame(data=rows,
                            columns=["tweet_id", "sentiment", "text"])

    def preprocess_data(self, data):
        return [" ".join(tokens) for tokens
                in self.preprocessor.pre_process_docs(data)]

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
