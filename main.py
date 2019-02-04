from runner import Runner
from data.data_loader import DataLoader
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from feature_extraction.automatic_features_counter import AutomaticFeaturesCounter
from feature_extraction.manual_features_counter import ManualFeaturesCounter
from etc.logger import Logger
import numpy as np

TERNARY = True

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

logger = Logger()
runner = Runner(logger=logger)
data_loader = DataLoader(preprocessor=preprocessor)

train, test = data_loader.get_train_test(ternary=TERNARY)

# mfc = ManualFeaturesCounter()
# afc = AutomaticFeaturesCounter()
#
# manual_train_features = mfc.get_features(train)
# manual_test_features = mfc.get_features(test)
#
# auto_train_features = afc.get_features(train)
# auto_test_features = afc.get_features(test)
#
# train_features = np.concatenate(
#     (manual_train_features, auto_train_features), axis=1)
# test_features = np.concatenate(
#     (manual_test_features, auto_test_features), axis=1)

logger.pre_setup(preprocessor="ekphrasis")

runner.run(train, test,
           ternary=TERNARY, model="elmo",
           use_embeddings=False)
