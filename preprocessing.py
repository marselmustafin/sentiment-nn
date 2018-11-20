from nltk.tokenize import TweetTokenizer
from nltk import SnowballStemmer
import html

class TextPreprocessor:
    def preprocess(self, text):
        text = self.__clean_text(text)
        tokens = self.__tokenize(text)
        stemmer = SnowballStemmer("english")
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        return stemmed_tokens

    def __tokenize(self, sent):
        tokenizer = TweetTokenizer(strip_handles=True,
                                   preserve_case=True, reduce_len=True)
        return tokenizer.tokenize(sent)

    def __clean_text(self, text):
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
