from nltk.tokenize import TweetTokenizer


class TextPreprocessor:
    def preprocess(self, text):
        tokenized_sent = self.__tokenize(text)
        return tokenized_sent

    def __tokenize(self, sent):
        tokenizer = TweetTokenizer(strip_handles=True,
                                   preserve_case=True, reduce_len=True)
        return tokenizer.tokenize(sent)
