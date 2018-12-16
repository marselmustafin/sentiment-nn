import numpy as np
import pandas as pd
from keras.layers import LSTM, Embedding, Dropout, Dense, Input, concatenate
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from embeddings.embedding_manager import EmbeddingManager
import ipdb


class BaselineModel:
    EMBEDDING_DIM = 50
    LSTM_OUT_DIM = 300

    def run(self, train, test, ternary=False, use_embeddings=True, features=None, test_features=None):
        tokenizer = Tokenizer(split=' ', filters="\n\t")
        tokenizer.fit_on_texts(train.text.values)
        vocab_size = len(tokenizer.word_index) + 1

        X_train, Y_train = self.features_targets(train, tokenizer)
        X_test, Y_test = self.features_targets(
            test, tokenizer, features_dim=X_train.shape[1])

        class_count = 3 if ternary else 2

        if use_embeddings:
            embedding_manager = EmbeddingManager()
            embedding_matrix = embedding_manager.get_embedding_matrix(
                tokenizer.word_index, self.EMBEDDING_DIM)
        else:
            embedding_matrix = None

        self.model = self.compile_model(
            vocab_size=vocab_size,
            input_dim=X_train.shape[1],
            class_count=class_count,
            embedding_matrix=embedding_matrix
        )

        earlystop = EarlyStopping(monitor='loss', min_delta=0.0001, patience=2,
                                  verbose=1, mode='auto')

        self.model.fit(
            [X_train, features],
            Y_train,
            callbacks=[earlystop],
            epochs=1,
            verbose=1)

        pred_classes = self.model.predict([X_test, test_features], verbose=1)

        ipdb.set_trace()

        self.print_results(pred_classes, Y_test, class_count=class_count)
        self.save_output_for_scoring(test.tweet_id, pred_classes)

    def compile_model(self, vocab_size=None, input_dim=None,
                      class_count=2, embedding_matrix=None):
        main_input = Input(shape=(input_dim,), name="main_input")
        features_input = Input(shape=(4,), name="features_input")

        # lstms = Sequential()

        # lstms.add(main_input)

        if embedding_matrix is not None:
            emb = Embedding(
                vocab_size,
                self.EMBEDDING_DIM,
                input_length=input_dim,
                weights=[embedding_matrix],
                trainable=False)(main_input)
        else:
            emb = Embedding(vocab_size, self.EMBEDDING_DIM,
                            input_length=input_dim)(main_input)

        drop = Dropout(0.2)(emb)
        lstm1 = LSTM(self.LSTM_OUT_DIM, return_sequences=True)(drop)
        lstm2 = LSTM(int(self.LSTM_OUT_DIM / 2), return_sequences=True)(lstm1)
        lstm3 = LSTM(int(self.LSTM_OUT_DIM / 4))(lstm2)

        lstms_with_features = concatenate([lstm3, features_input])

        final = Dense(3, activation='softmax')(lstms_with_features)

        model = Model(inputs=[main_input, features_input], outputs=final)

        model.compile(loss='categorical_crossentropy', optimizer='adam')

        print("%d CLASSIFYING MODEL", class_count)
        model.summary()

        return model

    def features_targets(self, dataframe, tokenizer, features_dim=None):
        X = tokenizer.texts_to_sequences(dataframe.text.values)
        X = pad_sequences(X, maxlen=features_dim)
        Y = pd.get_dummies(dataframe.sentiment).values

        return X, Y

    def save_output_for_scoring(self, test_ids, predictions):
        class_mappings = {0: "negative", 1: "neutral", 2: "positive"}
        results = [[tweet_id, class_mappings[sentiment]]
                   for tweet_id, sentiment in zip(test_ids, predictions)]
        results_pd = pd.DataFrame(data=results)
        results_pd.to_csv(
            path_or_buf="marsel.output",
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

        print(classification_report(test_classes,
                                    pred_classes, target_names=target_names))
