# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 23:25:24 2021

@author: Mustafa
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


url= 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
titles= ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

database= pd.read_csv(url, names=titles)

x= database.iloc[:,0:8].values
y= database.iloc[:,8:9].values

#for outlier and extreme values 
from sklearn.preprocessing import RobustScaler
rs= RobustScaler()
X=rs.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(X,y,test_size=0.35,random_state=0)




#training artificial neural network

print(x_train.shape)


import tensorflow as tf
from tensorflow import keras


model= tf.keras.models.Sequential([
    #tf.keras.layers.Flatten(input_shape=(8, 8)),
    tf.keras.layers.Dense(15,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20,activation='softmax')])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train,y_train, epochs=25)


# evaluate the keras model
_,accuracy_rate = model.evaluate(x_test, y_test)
print('Accuracy: %.2f' % (accuracy_rate*100))



# make probability predictions with the model
predictions = model.predict(x_test)
# round predictions 
rounded = [round(x_test[0]) for x_test in predictions]

from sklearn.metrics import mean_squared_error
MSE= mean_squared_error(y_test,rounded)
print('MSE : ',MSE)


liste= ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

# create a figure and axis 
for i in liste:
    fig, ax = plt.subplots() 
    # count the occurrence of each class 
    data = database[i].value_counts() 
    # get x and y data 
    points = data.index 
    frequency = data.values 
    # create bar chart 
    ax.bar(points, frequency) 
    # set title and labels 
    ax.set_title('Values of Plas') 
    ax.set_xlabel('Points') 
    ax.set_ylabel('Frequency')

#Analyses of result is finding the best iteration which is 25 epochs to Accuracy upper 70 percent
    
    
#This step is optimization of parameers depends on ANN. Unfortunatly this step is not run.
"""
#in order to best parameters
a= 10,11,12,13,14,15,18,20,22,23,28
b=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
ep=5,10,15,20,25,30

best_score=0
for i in a:
    for j in b:
            model= tf.keras.models.Sequential([
                #tf.keras.layers.Flatten(input_shape=(8, 8)),
                tf.keras.layers.Dense(i,activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(j,activation='softmax')])
            model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
            model.fit(x_train,y_train,epochs=20)
            _,accuracy_rate = model.evaluate(x_test, y_test)
            print('Accuracy: %.2f' % (accuracy_rate*100))
            if best_score<accuracy_rate:
                best_score=accuracy_rate
                bi=i
                bj=j
                print(bi, bj)
    

cik= input('Do you quit(q)')
if (cik== 'q'):
    print('break the program. . . ')
else:
    time.sleep(15)
"""
