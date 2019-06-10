import re
from string import punctuation
from nltk.corpus import stopwords


class NegationMarker():
    negations = """^(?:never|no|nothing|nowhere|noone|none|not|havent|hasnt|
                hadnt|cant|cannot|couldnt|shouldnt|wont|wouldnt|dont|doesnt|
                didnt|isnt|arent|aint)$|.*?n't"""
    context_stoppers = "^[.:;,)(!?\"]$|^but$"
    backoff_tag = "</?\w.*>"
    num = "\d+"

    def __init__(self):
        self.stoppers = re.compile(self.context_stoppers)
        self.negations = re.compile(self.negations)
        self.back_off_tag = re.compile(self.backoff_tag)
        self.num = re.compile(self.num)

    def mark_docs_negations(self, docs):
        marked_docs = [self.mark_negations(doc) for doc in docs]
        return marked_docs

    def mark_negations(self, doc):
        self.negated_context_counter = -1
        processed_tokens = [self.process_token(token) for token in doc.split()]

        return processed_tokens

    def process_token(self, token):
        if self.negated_context_counter < 0:
            if self.negations.match(token):
                self.negated_context_counter = 0
        else:
            if self.stoppers.match(token):
                self.negated_context_counter = -1
            else:
                if not (self.non_negotiable_part(token)
                        or self.negations.match(token)):
                    if self.negated_context_counter > 0:
                        token = token + "_NEG"
                    else:
                        token = token + "_NEGFIRST"

                self.negated_context_counter += 1

        return token

    def non_negotiable_part(self, token):
        return self.back_off_tag.match(token) \
            or token in punctuation \
            or self.num.match(token)
            # or token in set(stopwords.words("english"))
