# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 20:11:51 2021

@author: Mustafa
"""

#Type 1. Processing of csv files. It is including import the libraries, calling database, and visulization of data 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Step 1 and 2
url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
database= pd.read_csv(url)



#other step is transforming from something string values to numerical values becouse of LabelEncoder
#database= pd.read_excel('Iris_datasetr.xlsx')


from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
database.species= le.fit_transform(database.species)

#Step 3 dimension of database
print(database.shape)
print(database.size)

#Step 4 dimension of my database
x1= database.sepal_length
x2=database.sepal_width
axis= np.arange(len(x1))
plt.scatter(axis,x1,color='red')

plt.show()
plt.scatter(axis,x2,color='blue')
plt.show()

import seaborn as sns
sns.scatterplot(data=database)


#Type2. calling base sql database
import pandas as pd
import sqlalchemy

url= 'https://raw.githubusercontent.com/moneymanagerex/database/master/tables.sql'

engine= sqlalchemy.create_engine(url)
type_2_database= pd.read_sql_table('CHECKINGACCOUNT',engine)


#type 3 calling png files
url_png= 'https://image.flaticon.com/icons/png/512/29/29072.png'
image = plt.imread(url_png)

#type 4 calling video
import cv2
cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

#type 5 calling text file
text_url= 'https://raw.githubusercontent.com/Belval/TextRecognitionDataGenerator/master/requirements.txt'
file_text= open(text_url,'r')
file_text.close()
