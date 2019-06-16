# Computer Vision

## Introduction

Neural Networks for Computer Vision in PyTorch. This repository contains various image classification tasks on various data sets. 

## 1. Classyfying flowers into various species - image_classification.py

Used *transfer learning* on the *Oxford 102 Category flower dataset* to predict the species a flower belongs to. 

## Prerequistes

```
numpy - pip instll numpy or conda install numpy 
matplotlib - pip install matplotlib or conda install matplotlib
```
## Get Pytorch 

Pytorch - Visit https://pytorch.org/ and choose installation based on your machine's specific configuration.

## Data 

The dataset for this project is at http://www.robots.ox.ac.uk/~vgg/data/flowers/102/

## GPU vs CPU

It is adviced to run this on a GPU

## 2. Classying clothes to categories, FMNIST: The Fashion MNIST - FMNIST.py

Used a Convolutional Neural Network to classify clothes to their respective categories by taining it from scratch. 

## Prerequistes
```
numpy - pip install numpy if using pip or conda install numpy 
matplotlib - pip install matplotlib or conda install matplotlib
```
## Data

The dataset for this project need not be dowloaded as Pytorch comes with the dataset in the *datasets() class*. 
The curious ones can checkout the github page of the Fashion Mnist dataset https://github.com/zalandoresearch/fashion-mnist. 
It comes out of zolanda research and aims to be a replacement for benchmarking various Deep learning models. It's class 
distribution and train and test split are both the same as that of the famous MNIST digits dataset.

## 3. The legendary MNIST : MNIST.py

Here we classify digits ranging from 0 through 9 with the same convolutional neural network used for FMNIST. 


