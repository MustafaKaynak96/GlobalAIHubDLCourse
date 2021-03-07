# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 11:57:06 2021

@author: Mustafa
"""
1
#Importing packages

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


#load data CİFAR100 and description of datasets
#Training parameters
(X_train,y_train),(X_test,y_test)= datasets.cifar100.load_data()
X_train.shape

#Plt figure of one data of CIFAR100
plt.figure(figsize=(15,2))
plt.imshow(X_train[1])

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()


"""
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train,y_train, epochs=25)
"""

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# evaluate the keras model
_,accuracy_rate = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy_rate*100))



# make probability predictions with the model
predictions = model.predict(X_test)
# round predictions 
rounded = [round(x_test[0]) for x_test in predictions]

from sklearn.metrics import mean_squared_error
MSE= mean_squared_error(y_test,rounded)
print('MSE : ',MSE)

#Save of database
import pickle

with open('save_CNN','wb') as f:
    pickle.dump(model,f)

