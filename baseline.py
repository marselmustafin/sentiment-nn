import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, LSTM
from preprocessing.preprocessor import TextPreprocessor

TRAIN_SET = "data/train/twitter-2013train-A.tsv"
TEST_SET = "data/test/twitter-2013test-A.tsv"

train_data = pd.read_csv(TRAIN_SET, sep='\t',
                         header=None, names=["id", "sentiment", "text"],
                         usecols=["sentiment", "text"])

test_data = pd.read_csv(TEST_SET, sep='\t',
                        header=None, names=["id", "sentiment", "text", "unused "],
                        usecols=["sentiment", "text"])

preprocessor = TextPreprocessor()

train_data.text = train_data.text.apply(
    (lambda text: " ".join(preprocessor.preprocess(text))))

test_data.text = test_data.text.apply(
    (lambda text: " ".join(preprocessor.preprocess(text))))

tokenizer = Tokenizer(num_words=3000, split=' ', lower=False)
tokenizer.fit_on_texts(train_data.text.values)

X_train = tokenizer.texts_to_sequences(train_data.text.values)
X_train = pad_sequences(X_train)
Y_train = pd.get_dummies(train_data.sentiment).values

X_test = tokenizer.texts_to_sequences(test_data.text.values)
X_test = pad_sequences(X_test, maxlen=X_train.shape[1])
Y_test = pd.get_dummies(test_data.sentiment).values

EMBEDDING_DIM = 128
LSTM_OUT_DIM = 300

model = Sequential()
model.add(Embedding(3000, EMBEDDING_DIM, input_length=X_train.shape[1]))
model.add(LSTM(LSTM_OUT_DIM, dropout=0.2,
               recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(int(LSTM_OUT_DIM / 2), dropout=0.2,
               recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(3, dropout=0.2,
               recurrent_dropout=0.2, activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# class_weight = {0: 4.,
#                 1: 1.,
#                 2: 1.}

model.fit(X_train, Y_train, epochs=1, verbose=1)

model.summary()
test_classes = np.argmax(Y_test, axis=1)
pred_classes = model.predict_classes(X_test)
target_names = ['negative', 'neutral', 'positive']
print(classification_report(test_classes, pred_classes, target_names=target_names))

