from keras.layers import LSTM, Embedding, Dropout, Dense, Input, concatenate, \
    Bidirectional, GaussianNoise
from keras.models import Model
from layers.attention import Attention
from keras.optimizers import Adam
from keras.regularizers import l2

class BidirectionalAttention(object):
    LSTM_OUT_DIM = 150

    def compile(self, vocab_size=None, input_dim=None,
                class_count=2, embedding_matrix=None, embedding_dim=None,
                features_dim=None, dropout=0.2):

        main_input = Input(shape=(input_dim,), name="main_input")

        if embedding_matrix is not None:
            emb = Embedding(
                vocab_size,
                embedding_dim,
                input_length=input_dim,
                weights=[embedding_matrix],
                trainable=False)(main_input)
        else:
            emb = Embedding(vocab_size, embedding_dim,
                            input_length=input_dim)(main_input)

        gn = GaussianNoise(0.3)(emb)
        drop1 = Dropout(0.3, seed=123)(emb)
        lstm1 = Bidirectional(LSTM(self.LSTM_OUT_DIM,
                                   return_sequences=True))(drop1)
        drop2 = Dropout(0.3, seed=123)(lstm1)
        lstm2 = Bidirectional(LSTM(self.LSTM_OUT_DIM,
                                   return_sequences=True))(drop2)
        drop3 = Dropout(0.3, seed=123)(lstm2)
        attention = Attention()(drop3)
        drop4 = Dropout(0.5, seed=123)(attention)

        if features_dim is not None:
            features_input = \
                Input(shape=(features_dim,), name="features_input")

            lstms_with_features = concatenate([attention, features_input])
            dense = Dense(100, activation='relu')(lstms_with_features)
            final = Dense(class_count, activation='softmax')(dense)
            model = Model(inputs=[main_input, features_input], outputs=final)
        else:
            final = Dense(class_count, activation='softmax',
                          activity_regularizer=l2(0.0001))(drop4)
            model = Model(inputs=[main_input], outputs=final)

        model.compile(optimizer=Adam(clipnorm=1, lr=0.001),
                      loss='categorical_crossentropy', metrics=['accuracy'])

        return model
