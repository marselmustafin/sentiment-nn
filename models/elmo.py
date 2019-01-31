from keras.layers import LSTM, Dropout, Dense, Input, concatenate
from keras.models import Model
from layers.elmo_layer import ELMoEmbedding


class ElmoModel(object):
    LSTM_OUT_DIM = 1024

    def compile(self, input_dim=None, class_count=2,
                features_dim=None, index_word=None, dropout=0.2):

        main_input = Input(shape=(input_dim,), name="main_input")
        emb = ELMoEmbedding(idx2word=index_word,
                            output_mode="elmo", trainable=True)(main_input)

        drop = Dropout(dropout, seed=123)(emb)
        lstm1 = LSTM(self.LSTM_OUT_DIM, return_sequences=False)(drop)
        # lstm2 = LSTM(self.LSTM_OUT_DIM, return_sequences=True)(lstm1)
        # lstm3 = LSTM(self.LSTM_OUT_DIM)(lstm2)

        # dense = Dense(100, activation='relu')(lstms_with_features)

        final = Dense(class_count, activation='softmax')(lstm1)

        model = Model(inputs=[main_input], outputs=final)

        model.compile(loss='categorical_crossentropy', optimizer='adam')

        return model
