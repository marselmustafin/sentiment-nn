from runner import Runner
from data.data_loader import DataLoader
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from etc.logger import Logger
from feature_extraction.feature_extractor import FeatureExtractor
import random as rn
import numpy as np
import tensorflow as tf
import os

# add seeds for reproducibility
SEED = 1337
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(SEED)
rn.seed(SEED)
tf.set_random_seed(SEED)

TERNARY = True

# preprocessor for training/test
preprocessor = TextPreProcessor(
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time',
               'date', 'number'],
    annotate={'hashtag', 'allcaps', 'elongated', "repeated", 'emphasis',
              'censored'},
    fix_html=True,
    segmenter="twitter",
    corrector="twitter",
    unpack_hashtags=True,
    unpack_contractions=True,
    spell_correct_elong=False,
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[emoticons])

# preprocessor for features extracting
feature_preprocessor = TextPreProcessor(
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time',
               'date', 'number'],
    fix_html=True,
    segmenter='twitter',
    corrector='twitter',
    unpack_contractions=True,
    spell_correct_elong=False,
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[emoticons])

logger = Logger()
runner = Runner(logger=logger, ternary=TERNARY,
                model_type='baseline', use_embeddings=True)

logger.write('preprocessing: %s' % (True if preprocessor else False))

data_loader = DataLoader(preprocessor=preprocessor)
train, test = data_loader.get_train_test(ternary=TERNARY)
extra_train = data_loader.get_train(ternary=TERNARY, \
    paths=['data/ydata-ynacc-v1_0_expert_annotations_filt.tsv'])

feature_data_loader = DataLoader(preprocessor=feature_preprocessor)
feature_extractor = FeatureExtractor(data_loader=feature_data_loader,
                                     logger=logger)

train_feats, test_feats = feature_extractor.get_train_test_features(
    ternary=TERNARY,
    manual=True,
    auto=True,
    scaled=False)

runner.run(train, test, extra_train=extra_train)
