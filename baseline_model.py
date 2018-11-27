import numpy as np
import pandas as pd
from keras.layers import LSTM, Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import classification_report

from data_loader import DataLoader
import json


class BaselineModel:
    EMBEDDING_DIM = 128
    LSTM_OUT_DIM = 300

    def __init__(self):
        self.data_loader = DataLoader()

    def run(self, ternary=False):
        train, test = self.data_loader.get_train_test(ternary=ternary)

        tokenizer = Tokenizer(split=' ')
        tokenizer.fit_on_texts(train.text.values)
        vocab_size = len(tokenizer.word_index.keys()) + 1

        X_train, Y_train = self.features_targets(train, tokenizer)
        X_test, Y_test = self.features_targets(
            test, tokenizer, features_dim=X_train.shape[1])

        class_count = 3 if ternary else 2

        self.model = self.compile_model(
            vocab_size=vocab_size,
            input_dim=X_train.shape[1],
            class_count=class_count
        )

        self.model.fit(X_train, Y_train, epochs=1, verbose=1)

        self.print_results(X_test, Y_test, class_count=class_count)

    def print_results(self, X_test, Y_test, class_count=None):
        test_classes = np.argmax(Y_test, axis=1)
        pred_classes = self.model.predict_classes(X_test)
        target_names = ['negative', 'neutral', 'positive'] if class_count == 3 else [
            'negative', 'positive']

        print(classification_report(test_classes,
                                    pred_classes, target_names=target_names))

    def features_targets(self, dataframe, tokenizer, features_dim=None):
        X = tokenizer.texts_to_sequences(dataframe.text.values)
        X = pad_sequences(X, maxlen=features_dim)
        Y = pd.get_dummies(dataframe.sentiment).values

        return X, Y

    def compile_model(self, vocab_size=None, input_dim=None, class_count=2):
        model = Sequential()

        model.add(Embedding(vocab_size, self.EMBEDDING_DIM,
                            input_length=input_dim))
        model.add(LSTM(self.LSTM_OUT_DIM, dropout=0.2,
                       recurrent_dropout=0.2, return_sequences=True))
        model.add(LSTM(int(self.LSTM_OUT_DIM / 2), dropout=0.2,
                       recurrent_dropout=0.2, return_sequences=True))
        model.add(LSTM(class_count, dropout=0.2,
                       recurrent_dropout=0.2, activation="softmax"))

        model.compile(loss='categorical_crossentropy', optimizer='adam')

        print("%d CLASSIFYING MODEL", class_count)
        model.summary()

        return model


baseline = BaselineModel()
baseline.run()
