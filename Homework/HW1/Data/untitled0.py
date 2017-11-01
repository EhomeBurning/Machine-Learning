#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 14:39:25 2017

@author: liuchangbai
"""

import numpy as np
from PIL import Image


################################# Part A ######################################
# Load Image
image = Image.open('harvey-saturday-goes7am.jpg')
grey = image.convert("L")

# Image to array 
X = np.asarray(grey)

# k value
k_list = [2,10,40]
result_list = []
partb_list = []

for k in k_list:
    # SVD
    U,s,Vt = np.linalg.svd(X, full_matrices = False)
    s[k:] = 0
    S = np.diag(s)
    X_app = np.dot(np.dot(U,S), Vt)
    
    # Show and save approximate_image 
    img = Image.fromarray(X_app)
    if(img.mode != 'RGB'):
        img = img.convert('RGB')
    
    img.save(str(k)+'.jpg')
    
    # calculate ||X-X_app||f / ||X||f
    temp = X - X_app
    result = np.linalg.norm(temp,'fro') / np.linalg.norm(X,'fro')
    result_list.append(result)
    
################################# Part B ######################################
    # How many numbers do you need to describe the approximation
    left = U[:,:k]
    left_row = left.shape[0]
    left_column = left.shape[1]
    num_left = left_row * left_column
    
    right = Vt[:k,:]
    right_row = right.shape[0]
    right_column = right.shape[1]
    num_right = right_row * right_column
    
    partb = num_left + num_right + k
    partb_list.append(partb)
    
    
    
    

print(result_list)
print(partb_list)













