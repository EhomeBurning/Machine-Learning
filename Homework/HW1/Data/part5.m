clc
clear all 
close all

X = double(rgb2gray(imread('harvey-saturday-goes7am.jpg')))
[U,S,V] = svd(X)

k = rank(X)






