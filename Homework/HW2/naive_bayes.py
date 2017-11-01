#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 13:56:53 2017

@author: liuchangbai
"""

import csv
import os
import numpy as np
import pandas as pd
import nb


label_buf = []
token_list = []
word_list = []

os.chdir("/Users/liuchangbai/Desktop/courses/Machine-Learning/Homework/HW2")

token_path = os.path.expanduser('./spam_classification/TOKENS_LIST')
with open(token_path,newline='') as token:
    reader = csv.reader(token, delimiter=' ')
    for row in reader:
        token_list.append(row)

for i in token_list:
    word_list.append(i[1])


train_path = os.path.expanduser('./spam_classification/SPARSE.TRAIN')
with open(train_path, newline='') as train:
    reader = csv.reader(train, delimiter=' ')
    for row in reader:
        label_buf.append(int(row[0]))
label = np.asarray(label_buf,dtype=int)


nd = len(label)
nw = len(token_list)
count_d_w = np.zeros([nd,nw],dtype=int)
with open(train_path, newline='') as train:
    reader = csv.reader(train, delimiter=' ')
    for d_id, row in enumerate(reader):
        current_email = csv.reader(row[2:-1],delimiter=':')
        for rows in current_email:
            w_id = int(rows[0])
            count = int(rows[1])
            count_d_w[d_id][w_id-1] = count

df_train = pd.DataFrame(count_d_w, columns = [word_list])
df_train["label"] = pd.Series(label)



# classify the test dataset
# read the test dataset
label_test_buf = list()
test_path = os.path.expanduser('./spam_classification/SPARSE.TEST')
with open(test_path, newline='') as test:
    reader = csv.reader(test, delimiter=' ')
    for row in reader:
        label_test_buf.append(int(row[0]))
label_test = np.asarray(label_test_buf,dtype=int)

nd_test = len(label_test)
count_d_w_test = np.zeros([nd_test,nw],dtype=int)
with open(test_path, newline='') as test:
    reader = csv.reader(test, delimiter=' ')
    for d_id, row in enumerate(reader):
        current_email = csv.reader(row[2:-1],delimiter=':')
        for rows in current_email:
            w_id = int(rows[0])
            count = int(rows[1])
            count_d_w_test[d_id][w_id-1] = count



    

df_test = pd.DataFrame(count_d_w_test)
nb_model = nb.train(df_train)
nb_predictions = nb.test(nb_model, df_test)
y = pd.Series(label_test)
nb_error = nb.compute_error(y, nb_predictions)
print('NB Test error: {}'.format(nb_error))



words = nb.k_most_indicative_words(5, nb_model.to_dataframe().iloc[:,:-1])
print('The {} most spam-worthy words are: {}'.format(len(words), words))





















