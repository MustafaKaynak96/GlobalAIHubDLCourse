# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 22:44:38 2021

@author: Mustafa
"""



import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing import image
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Model
import timeit

import warnings
warnings.filterwarnings('ignore')

batch_size = 128
num_classes = 10
epochs = 2

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

K.clear_session()
start = timeit.default_timer()   
model = Sequential()
model.add(Conv2D(8, kernel_size=(9, 9), activation='relu', input_shape=input_shape))
model.add(Conv2D(16, (9, 9), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
end = timeit.default_timer()
print("Time Taken to run the model:",end - start, "seconds")

#3 3 result
"""Epoch 1/2
469/469 [==============================] - 98s 197ms/step - loss: 2.3030 - accuracy: 0.1073 - val_loss: 2.2766 - val_accuracy: 0.1850: 42s - loss: 2.3084 - accuracy: 0.0942 - ETA: 9s - loss: 2.3039 - accuracy: 0.1048
Epoch 2/2
469/469 [==============================] - 75s 159ms/step - loss: 2.2667 - accuracy: 0.2020 - val_loss: 2.2338 - val_accuracy: 0.2283
Time Taken to run the model: 173.82527239999945 seconds
"""
#9 9 result
"""
Epoch 1/2
469/469 [==============================] - 108s 227ms/step - loss: 2.3043 - accuracy: 0.1042 - val_loss: 2.2833 - val_accuracy: 0.1641
Epoch 2/2
469/469 [==============================] - 108s 231ms/step - loss: 2.2721 - accuracy: 0.1924 - val_loss: 2.2387 - val_accuracy: 0.2752
Time Taken to run the model: 215.87010139999984 seconds
"""
# 9x9 its took a long time, but the accuracy increased