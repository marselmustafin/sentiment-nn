import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from embeddings.embedding_manager import EmbeddingManager
from models.baseline_with_features import BaselineWithFeatures
from models.elmo import ElmoModel
from models.bidirectional_attention import BidirectionalAttention
from kutilities.helpers.data_preparation import get_class_weights2, \
     onehot_to_categories

class Runner:
    def __init__(self, logger, ternary=False, epochs=20,
                 embedding_dim=50, batch_size=32, validation_split=0.1,
                 dropout=0.5, model_type=None, use_embeddings=False):
        assert model_type in ["elmo", "baseline", "bid_attent"]
        self.ternary = ternary
        self.epochs = epochs
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.dropout = dropout
        self.model_type = model_type
        self.use_embeddings = use_embeddings
        self.logger = logger
        self.tokenizer = Tokenizer(split=' ', filters="\n\t")

    def run(self, train, test, features=None, test_features=None, extra_train=None):
        self.tokenizer.fit_on_texts(train.text.values)

        features_dim = features.shape[1] if features is not None else None

        X_train, Y_train = self.get_features_targets(train)
        X_test, Y_test = self.get_features_targets(
            test, features_dim=X_train.shape[1])

        if extra_train is not None:
            self.tokenizer.fit_on_texts(extra_train.text.values)
            X_extra_train, Y_extra_train \
                = self.get_features_targets(extra_train)
            if X_extra_train.shape[1] > X_train.shape[1]:
                X_train = pad_sequences(X_train, maxlen=X_extra_train.shape[1])
                X_test = pad_sequences(X_test, maxlen=X_extra_train.shape[1])

        vocab_size = len(self.tokenizer.word_index) + 1

        class_count = 3 if self.ternary else 2

        embedding_matrix = None

        if self.use_embeddings:
            embedding_manager = EmbeddingManager()
            embedding_matrix = embedding_manager.get_embedding_matrix(
                self.tokenizer.word_index, self.embedding_dim)

        base_model_params = {
            'input_dim': X_train.shape[1],
            'class_count': class_count,
            'features_dim': features_dim,
            'dropout': self.dropout
        }

        if self.model_type == "elmo":
            params = {
                **base_model_params,
                'index_word': self.tokenizer.index_word
            }
            self.model = ElmoModel().compile(**params)
        elif self.model_type == "bid_attent":
            params = {
                **base_model_params,
                'vocab_size': vocab_size,
                'embedding_matrix': embedding_matrix,
                'embedding_dim': self.embedding_dim
            }
            self.model = BidirectionalAttention().compile(**params)
        else:
            params = {
                **base_model_params,
                'vocab_size': vocab_size,
                'embedding_matrix': embedding_matrix,
                'embedding_dim': self.embedding_dim
            }
            self.model = BaselineWithFeatures().compile(**params)

        self.logger.setup(
            ternary=self.ternary,
            embeddings=self.use_embeddings,
            train_set=X_train,
            test_set=X_test,
            vocab_size=vocab_size,
            epochs=self.epochs,
            batch_size=self.batch_size,
            dropout=self.dropout,
            extra_train=extra_train is not None
        )

        self.model.summary(print_fn=self.logger.write)

        fit_params = {
            'batch_size': self.batch_size,
            'callbacks': self.get_callbacks(),
            'epochs': self.epochs,
            'validation_split': self.validation_split,
            'verbose': 1,
            'class_weight': get_class_weights2(onehot_to_categories(Y_train),
                                               smooth_factor=0)
        }

        if features is not None:
            self.model.fit([X_train, features], Y_train, **fit_params)
            pred_classes = self.model.predict(
                [X_test, test_features], verbose=1)
        else:
            if extra_train is not None:
                training = self.model.fit(
                    X_extra_train, Y_extra_train, **fit_params)
                self.logger.write_history(training)
            training = self.model.fit(X_train, Y_train, **fit_params)
            self.logger.write_history(training)

            pred_classes = self.model.predict(X_test, verbose=1)

        pred_classes = pred_classes.argmax(axis=1)

        self.print_results(pred_classes, Y_test, class_count=class_count)
        self.save_output_for_scoring(test.tweet_id, pred_classes)

    def get_callbacks(self):
        earlystop = EarlyStopping(monitor='loss', min_delta=0.01, patience=2,
                                  verbose=1, mode='auto')
        # plotting = PlottingCallback(grid_ranges=(0.5, 0.75), height=5,
        #                     benchmarks={"SE17": 0.681})
        return [earlystop]

    def get_features_targets(self, dataframe, features_dim=None):
        X = self.tokenizer.texts_to_sequences(dataframe.text.values)
        X = pad_sequences(X, maxlen=features_dim)
        Y = pd.get_dummies(dataframe.sentiment).values

        return X, Y

    def save_output_for_scoring(self, test_ids, predictions):
        class_mappings = {0: "negative", 1: "neutral", 2: "positive"}
        results = [[tweet_id, class_mappings[sentiment]]
                   for tweet_id, sentiment in zip(test_ids, predictions)]
        results_pd = pd.DataFrame(data=results)
        results_pd.to_csv(
            path_or_buf=self.logger.dir + "prediction.output",
            sep="\t",
            header=None,
            index=None)

    def print_results(self, pred_classes, Y_test, class_count=None):
        test_classes = np.argmax(Y_test, axis=1)
        target_names = [
            'negative',
            'neutral',
            'positive'] if class_count == 3 else [
            'negative',
            'positive']
        report = classification_report(test_classes, pred_classes,
                                       target_names=target_names, digits=3)
        self.logger.write(report)
        self.logger.write(np.array2string(confusion_matrix(test_classes,
                                                           pred_classes)))
        print(report)
