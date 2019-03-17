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


class Runner:
    EMBEDDING_DIM = 300
    EPOCHS = 10
    BATCH_SIZE = 32
    DROPOUT = 0.5

    def __init__(self, logger):
        self.logger = logger
        self.tokenizer = Tokenizer(split=' ', filters="\n\t")

    def run(self, train, test, ternary=False, use_embeddings=False,
            features=None, test_features=None, model=None, extra_train=None):
        self.tokenizer.fit_on_texts(train.text.values)

        if extra_train is not None:
            self.tokenizer.fit_on_texts(extra_train.text.values)

        vocab_size = len(self.tokenizer.word_index) + 1

        features_dim = features.shape[1] if features is not None else None

        X_train, Y_train = self.get_features_targets(train)
        X_test, Y_test = self.get_features_targets(
            test, features_dim=X_train.shape[1])

        class_count = 3 if ternary else 2

        if use_embeddings:
            embedding_manager = EmbeddingManager()
            embedding_matrix = embedding_manager.get_embedding_matrix(
                self.tokenizer.word_index, self.EMBEDDING_DIM)
        else:
            embedding_matrix = None

        if model == "elmo":
            self.model = ElmoModel().compile(
                input_dim=X_train.shape[1],
                class_count=class_count,
                features_dim=features_dim,
                index_word=self.tokenizer.index_word,
                dropout=self.DROPOUT
            )
        elif model == "bid_attent":
            self.model = BidirectionalAttention().compile(
                vocab_size=vocab_size,
                input_dim=X_train.shape[1],
                class_count=class_count,
                features_dim=features_dim,
                embedding_matrix=embedding_matrix,
                embedding_dim=self.EMBEDDING_DIM,
                dropout=self.DROPOUT
            )
        else:
            self.model = BaselineWithFeatures().compile(
                vocab_size=vocab_size,
                input_dim=X_train.shape[1],
                class_count=class_count,
                features_dim=features_dim,
                embedding_matrix=embedding_matrix,
                embedding_dim=self.EMBEDDING_DIM,
                dropout=self.DROPOUT
            )

        earlystop = EarlyStopping(monitor='loss', min_delta=0.01, patience=2,
                                  verbose=1, mode='auto')

        self.logger.setup(
            ternary=ternary,
            embeddings=use_embeddings,
            train_set=X_train,
            test_set=X_test,
            vocab_size=vocab_size,
            earlystop=earlystop,
            epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            dropout=self.DROPOUT,
            extra_train=extra_train is not None
        )

        self.model.summary(print_fn=self.logger.write)

        if features is not None:
            self.model.fit(
                [X_train, features],
                Y_train,
                batch_size=self.BATCH_SIZE,
                callbacks=[earlystop],
                epochs=self.EPOCHS,
                verbose=1)

            pred_classes = self.model.predict(
                [X_test, test_features], verbose=1)
        else:
            self.model.fit(
                X_train,
                Y_train,
                batch_size=self.BATCH_SIZE,
                callbacks=[earlystop],
                epochs=self.EPOCHS,
                verbose=1)
            if extra_train is not None:
                X_extra_train, Y_extra_train \
                    = self.get_features_targets(extra_train)
                self.model.fit(
                    X_extra_train,
                    Y_extra_train,
                    batch_size=self.BATCH_SIZE,
                    callbacks=[earlystop],
                    epochs=self.EPOCHS,
                    verbose=1)

            pred_classes = self.model.predict(X_test, verbose=1)

        pred_classes = pred_classes.argmax(axis=1)

        self.print_results(pred_classes, Y_test, class_count=class_count)
        self.save_output_for_scoring(test.tweet_id, pred_classes)

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
