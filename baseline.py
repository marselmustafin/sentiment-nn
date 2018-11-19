import re

import ipdb
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

from preprocessing import TextPreprocessor

INPUT_FILE = "data/SemEval2017-task4-dev.subtask-A.english.INPUT.txt"

data = pd.read_csv(INPUT_FILE, sep='\t',
                   header=None, names=["id", "sentiment", "text", "unused"],
                   usecols=["sentiment", "text"])

data.text = data.text.apply(
    (lambda text: " ".join(TextPreprocessor(text).preprocess())))

tokenizer = Tokenizer(num_words=2500, split=' ', lower=True)
tokenizer.fit_on_texts(data.text.values)
X = tokenizer.texts_to_sequences(data.text.values)
X = pad_sequences(X)

embed_dim = 128

model = Sequential()
model.add(Embedding(2500, embed_dim,
                    input_length=X.shape[1], dropout=0.2))
model.add(LSTM(200, dropout_U=0.2, dropout_W=0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

Y = pd.get_dummies(data['sentiment']).values
X_train, X_valid, Y_train, Y_valid = train_test_split(
    X, Y, test_size=0.20, random_state=36)

model.fit(X_train, Y_train, batch_size=32, epochs=1, verbose=2)

# Measuring score and accuracy on validation set

score, acc = model.evaluate(X_valid, Y_valid, verbose=2, batch_size=32)
print("Logloss score: %.2f" % (score))
print("Validation set Accuracy: %.2f" % (acc))
