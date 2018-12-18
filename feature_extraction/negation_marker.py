import re
from string import punctuation

class NegationMarker():
    NEGATIONS = """^(?:never|no|nothing|nowhere|noone|none|not|havent|hasnt|
                hadnt|cant|cannot|couldnt|shouldnt|wont|wouldnt|dont|doesnt|
                didnt|isnt|arent|aint)$|.*?n't"""
    CONTEXT_STOPPERS = "^[.:;,)(!?\"]$|^but$"
    BACKOFF_TAG = "</?\w.*>"
    NUM = "\d+"

    def __init__(self):
        self.stoppers = re.compile(self.CONTEXT_STOPPERS)
        self.negations = re.compile(self.NEGATIONS)
        self.back_off_tag = re.compile(self.BACKOFF_TAG)
        self.num = re.compile(self.NUM)

    def mark_docs_negations(self, docs):
        marked_docs = [self.mark_negations(doc) for doc in docs]
        return marked_docs

    def mark_negations(self, doc):
        self.negated_context = False
        processed_tokens = [self.process_token(token) for token in doc.split()]

        return processed_tokens

    def process_token(self, token):
        if not self.negated_context:
            if self.negations.match(token):
                self.negated_context = True
        else:
            if self.stoppers.match(token):
                self.negated_context = False
            elif not (self.is_back_off_punct_or_num(token) or self.negations.match(token)):
                token = token + "_NEG"

        return token

    def is_back_off_punct_or_num(self, token):
        return self.back_off_tag.match(token) or token in punctuation or self.num.match(token)
