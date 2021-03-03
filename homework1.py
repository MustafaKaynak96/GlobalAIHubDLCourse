# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 21:15:56 2021

@author: Mustafa
"""


#Explain your work

#Question 1
a=10
#it will be print on screen as same values depands on a object
for x in range(a):
	print(a)

print('............................')
#this loop will going the zero to nine integer values sequently
for x in range(a):
    print(x)

print('.............................')
#we can assign variables as object
a= int(input('enter a values'))
for x in range(a):
    print(x)
#we can save such as dataframe variables
#we use pandas library and append commant to DataFrame. Other thing, we can assign values such as list    
import pandas as pd
a= 10
Data_frame= []
for x in range(a):
    print(x)
    Data_frame.append(x)
dataframe = pd.DataFrame(Data_frame)

    