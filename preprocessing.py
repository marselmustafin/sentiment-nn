from nltk.tokenize import TweetTokenizer


class TextPreprocessor:
    def __init__(self, text):
        self.__text = text

    def preprocess(self):
        tokenized_sent = self.__tokenize(self.__text)
        return tokenized_sent

    def __tokenize(self, sent):
        tokenizer = TweetTokenizer(strip_handles=True,
                                   preserve_case=True, reduce_len=True)
        return tokenizer.tokenize(sent)
