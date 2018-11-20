import re
from keras import backend as K
import ipdb
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

from preprocessing import TextPreprocessor

TRAIN_SET = "data/train/twitter-2013train-A.tsv"
TEST_SET = "data/train/twitter-2013test-A.tsv"


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


train_data = pd.read_csv(TRAIN_SET, sep='\t',
                         header=None, names=["id", "sentiment", "text"],
                         usecols=["sentiment", "text"])

test_data = pd.read_csv(TRAIN_SET, sep='\t',
                        header=None, names=["id", "sentiment", "text"],
                        usecols=["sentiment", "text"])

preprocessor = TextPreprocessor()

train_data.text = train_data.text.apply(
    (lambda text: " ".join(preprocessor.preprocess(text))))

test_data.text = test_data.text.apply(
    (lambda text: " ".join(preprocessor.preprocess(text))))

train_tokenizer = Tokenizer(num_words=2500, split=' ', lower=True)
train_tokenizer.fit_on_texts(train_data.text.values)
X_train = train_tokenizer.texts_to_sequences(train_data.text.values)
X_train = pad_sequences(X_train)
Y_train = pd.get_dummies(train_data.sentiment).values

test_tokenizer = Tokenizer(num_words=2500, split=' ', lower=True)
test_tokenizer.fit_on_texts(test_data.text.values)
X_test = test_tokenizer.texts_to_sequences(test_data.text.values)
X_test = pad_sequences(X_test)
Y_test = pd.get_dummies(test_data.sentiment).values

embed_dim = 128
lstm_out = 300
batch_size = 32

# Buidling the LSTM network

model = Sequential()
model.add(Embedding(2500, embed_dim,
                    input_length=X_train.shape[1], dropout=0.1))
model.add(LSTM(lstm_out, dropout=0.1,
               recurrent_dropout=0.1, return_sequences=True))
model.add(LSTM(lstm_out, dropout=0.1,
               recurrent_dropout=0.1, return_sequences=True))
model.add(LSTM(lstm_out, dropout=0.1,
               recurrent_dropout=0.1))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy', precision, recall])

# Here we train the Network.

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1,  verbose=5)

# Measuring score and accuracy on validation set

score, acc, precision, recall = model.evaluate(
    X_test, Y_test, verbose=2, batch_size=batch_size)
print("Logloss score: %.2f" % (score))
print("Validation set Accuracy: %.2f" % (acc))
print("Validation set Precision: %.2f" % (precision))
print("Validation set Recall: %.2f" % (recall))
