from baseline_model import BaselineModel
from data.data_loader import DataLoader
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from feature_extraction.automatic_features_counter import AutomaticFeaturesCounter
from feature_extraction.manual_features_counter import ManualFeaturesCounter
import numpy as np


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
data_loader = DataLoader(preprocessor=preprocessor)

train, test = data_loader.get_train_test(ternary=True)

mfc = ManualFeaturesCounter()
afc = AutomaticFeaturesCounter()

manual_train_features = mfc.get_features(train)
manual_test_features = mfc.get_features(test)

auto_train_features = afc.get_features(train)
auto_test_features = afc.get_features(test)

train_features = np.concatenate((manual_train_features, auto_train_features), axis=1)
test_features = np.concatenate((manual_test_features, auto_test_features), axis=1)

model.run(train, test,
          ternary=True,
          use_embeddings=False,
          features=train_features,
          test_features=test_features)
