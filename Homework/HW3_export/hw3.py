#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:29:49 2017

@author: liuchangbai
"""


import pandas as pd
import numpy as np
import os,random

os.chdir("/Users/liuchangbai/Desktop/courses/Machine-Learning/Homework/HW3_export")

data = pd.read_csv("diabetes_scale.csv", sep = ",", names = ['label', 'feature1', 'feature2','feature3',
                                                             'feature4','feature5','feature6','feature7','feature8'])

test = data[500:768]
data = data[0:500]


# cross validation 
y = data['label']
x = data[['feature1', 'feature2','feature3','feature4','feature5','feature6','feature7','feature8']]

y_final = test['label']
x_final = test[['feature1', 'feature2','feature3','feature4','feature5','feature6','feature7','feature8']]

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=43)


# C value 
c_list = np.linspace(0.1, 2, 20)
score_dict = {}

for c_value in c_list:
    # Support Vector Machine 
    from sklearn import svm
    clf = svm.SVC(C = c_value)
    
    # fit
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    
    # get prediction score 
    from sklearn import metrics
    score = metrics.accuracy_score(y_test,y_pred)
    
    score_dict[c_value] = score


#c_value = 1.7
#clf = svm.SVC(C = c_value)

y_predict = clf.predict(x_final)
soft_score = metrics.accuracy_score(y_final, y_predict)



# Hard Margin
hdm = svm.SVC(C = 1* np.exp(6))
hdm.fit(x_train, y_train)

y_pred = hdm.predict(x_final)

# get prediction score 

print(metrics.accuracy_score(y_final,y_pred))



