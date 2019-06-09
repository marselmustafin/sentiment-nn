from keras.layers import LSTM, Dropout, Dense, Input, Concatenate, \
    Convolution1D, MaxPooling1D, Flatten
from keras.models import Model
from layers.elmo_layer import ELMoEmbedding


class ElmoModel(object):
    LSTM_OUT_DIM = 1024
    KERNEL_SIZES = [2, 3, 5]
    DROPOUT = 0.4
    HIDDEN_DIMS = 50
    NUM_FILTERS = 64

    def compile(self, input_dim=None, class_count=2,
                features_dim=None, index_word=None, dropout=0.2):

        main_input = Input(shape=(input_dim,), name="main_input")
        emb = ELMoEmbedding(idx2word=index_word,
                            output_mode="elmo", trainable=True)(main_input)

        conv_blocks = []
        for kernel_size in self.KERNEL_SIZES:
            conv = Convolution1D(filters=self.NUM_FILTERS,
                                 kernel_size=kernel_size,
                                 padding="valid",
                                 activation="relu",
                                 strides=1)(emb)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)

        conv_conc = Concatenate()(conv_blocks)

        drop = Dropout(self.DROPOUT, seed=123)(conv_conc)

        final = Dense(class_count, activation='softmax')(drop)

        model = Model(inputs=[main_input], outputs=final)

        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])

        return model
