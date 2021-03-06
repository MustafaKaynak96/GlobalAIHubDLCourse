# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 22:25:23 2021

@author: Mustafa
"""



import re
import numpy as np
import pandas as pd 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.datasets import imdb

from keras.utils.np_utils import to_categorical

import warnings
warnings.filterwarnings('ignore')
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


max_features = 1000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

# save np.load
#np_load_old = np.load

# modify the default parameters of np.load
#np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

#np.load = np_load_old

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


model = Sequential()
model.add(Embedding(max_features, 8))
model.add(LSTM(8, dropout=0.0, recurrent_dropout=0.0))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

#1.step result batch size= 32
"""
782/782 [==============================] - 44s 51ms/step - loss: 0.6065 - accuracy: 0.6528 - val_loss: 0.4092 - val_accuracy: 0.8108
782/782 [==============================] - 7s 9ms/step - loss: 0.4092 - accuracy: 0.8108
Test score: 0.40918999910354614
Test accuracy: 0.8107600212097168

loss: 0.4092 - accuracy: 0.8108
Test score: 0.40918999910354614
Test accuracy: 0.8107600212097168
"""

#1.step result batch_size= 64
"""loss: 0.4347 - accuracy: 0.8061
Test score: 0.43470534682273865
Test accuracy: 0.8060799837112427"""

#These batch size of iterations are closer themselves

