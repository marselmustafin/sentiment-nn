from keras.layers import LSTM, Embedding, Dropout, Dense, Input, concatenate
from keras.models import Model


class BaselineWithFeatures(object):
    EMBEDDING_DIM = 50
    LSTM_OUT_DIM = 300

    def compile(self, vocab_size=None, input_dim=None,
                class_count=2, embedding_matrix=None, features_dim=None):

        main_input = Input(shape=(input_dim,), name="main_input")
        features_input = Input(shape=(features_dim,), name="features_input")

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

        drop = Dropout(0.2, seed=123)(emb)
        lstm1 = LSTM(self.LSTM_OUT_DIM, return_sequences=True)(drop)
        lstm2 = LSTM(int(self.LSTM_OUT_DIM), return_sequences=True)(lstm1)
        lstm3 = LSTM(int(self.LSTM_OUT_DIM))(lstm2)

        lstms_with_features = concatenate([lstm3, features_input])

        dense = Dense(100, activation='relu')(lstms_with_features)

        final = Dense(class_count, activation='softmax')(dense)

        model = Model(inputs=[main_input, features_input], outputs=final)

        model.compile(loss='categorical_crossentropy', optimizer='adam')

        print("%d CLASSIFYING MODEL", class_count)
        model.summary()

        return model
