from runner import Runner
from data.data_loader import DataLoader
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from etc.logger import Logger
from feature_extraction.feature_extractor import FeatureExtractor

TERNARY = True

preprocessor = TextPreProcessor(
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time',
               'date', 'number'],
    annotate={"hashtag", "allcaps", "elongated", "repeated", 'emphasis',
              'censored'},
    fix_html=True,
    segmenter="twitter",
    corrector="twitter",
    unpack_hashtags=True,
    unpack_contractions=True,
    spell_correct_elong=False,
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[emoticons])

feature_preprocessor = TextPreProcessor(
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time',
               'date', 'number'],
    fix_html=True,
    segmenter="twitter",
    corrector="twitter",
    unpack_contractions=True,
    spell_correct_elong=False,
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[emoticons])

logger = Logger()
runner = Runner(logger=logger)

logger.write("preprocessing: %s" % (True if preprocessor else False))

data_loader = DataLoader(preprocessor=preprocessor)
feature_data_loader = DataLoader(preprocessor=feature_preprocessor)

train, test = data_loader.get_train_test(ternary=TERNARY)

f_train, f_test = feature_data_loader.get_train_test(ternary=TERNARY)

feature_extractor = FeatureExtractor(logger=logger)

tr_feats = feature_extractor.get_features(f_train)
te_feats = feature_extractor.get_features(f_test)

runner.run(train, test,
           ternary=TERNARY,
           features=tr_feats,
           test_features=te_feats,
           use_embeddings=True)
