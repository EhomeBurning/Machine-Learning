#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 19:52:44 2017

@author: liuchangbai
"""

import os
import numpy as np
import scipy.io as sio
from sklearn import cross_validation, neighbors
import operator

os.chdir("/Users/liuchangbai/Desktop/courses/Machine-Learning/Homework/HW2")

mnist_data = sio.loadmat('mnist_data.mat')

train_array = np.array(mnist_data['train'])
test_array = np.array(mnist_data['test'])
  
random_numbers = np.random.choice(10000,50)
test_data = test_array[random_numbers]



def KNN(feature, predict, k):
    X = feature
    y = predict
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size = 0.2)
    
    classifier = neighbors.KNeighborsClassifier()
    classifier.fit(X_train,y_train)

    accuracy = classifier.score(X_test, y_test)
    return accuracy
    

def l1distance(instance1, instance2, length):
    distance = 0
    for x in range(1,length+1):
        distance += abs(instance1[x]-instance2[x])
    return distance

 def l2distance(instance1, instance2, length):
    distance = 0
    for x in range(1,length+1):
        distance+=pow((instance1[x]-instance2[x]),2)
    return distance
    
def getNeighbors(trainingSet, testInstance, k):
    distance=[]
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distance.append((trainingSet[x],dist))
    distance.sort(key=operator.itemgetter(1))
    neighbors=[]
    for x in range(k):
        neighbors.append(distance[x][0])  
    classVotes={}
    for x in range(len(neighbors)):
        response = neighbors[x][0]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct=0
    for x in range(len(testSet)):
        if testSet[x][0]==predictions[x]:
            correct+=1
    return (correct/float(len(testSet))) *100


K = 5 #{1,5,9,13}
predictions = []
    
for x in range(len(test_data)):
    neighbors = getNeighbors(train_array, test_data[x], K)
    result = getResponse(neighbors)
    predictions.append(result)

accuracy = getAccuracy(test_data, predictions)










    

