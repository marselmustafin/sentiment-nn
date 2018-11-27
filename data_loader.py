from preprocessing.preprocessor import TextPreprocessor
import pandas as pd


class DataLoader:
    TRAIN_DATA_PATH = "data/train/original/twitter-2013train-A.tsv"
    TEST_DATA_PATH = "data/test/original/twitter-2013test-A.tsv"

    def __init__(self):
        self.train_data = self.read_data(self.TRAIN_DATA_PATH)
        self.test_data = self.read_data(self.TEST_DATA_PATH)
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

    def read_data(self, filename):
        return pd.read_csv(filename, sep='\t', header=None,
                           names=["id", "sentiment", "text"],
                           usecols=["sentiment", "text"])

    def preprocess_data(self, data):
        return data.apply(
            (lambda text: " ".join(self.preprocessor.preprocess(text))))
