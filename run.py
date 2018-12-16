from baseline_model import BaselineModel
from data.data_loader import DataLoader
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from preprocessing.preprocessor import TextPreprocessor
from feature_extraction.feature_extractor import FeatureExtractor

import ipdb


preprocessor = TextPreProcessor(
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time',
               'url',
               'date', 'number'],
    annotate={"hashtag", "allcaps", "elongated", "repeated",
              'emphasis',
              'censored'},
    fix_html=True,
    segmenter="twitter",
    corrector="twitter",
    unpack_hashtags=True,
    unpack_contractions=True,
    spell_correct_elong=False,
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[emoticons])

model = BaselineModel()
data_loader = DataLoader(preprocessor)

train, test = data_loader.get_train_test(ternary=True)

fe = FeatureExtractor()

features = fe.count_features(train)
test_features = fe.count_features(test)

model.run(train, test, ternary=True, use_embeddings=True, features=features, test_features=test_features)
ipdb.set_trace()
