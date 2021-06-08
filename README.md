## Introduction
This repository presents a machine learning solution for classifying Fashion MNIST dataset.

Fashion MNIST dataset contains 70,000 examples of 28x28 grayscale images of garment, each labelled with one of the ten
categories. Examples are split into two groups, test set with 10,000 elements and training set with 60,000 elements.

The goal is to find an algorithm that will predict labels of images in test set, after being shown the training set
with labels. Quality of an algorithm can be measured in terms of its accuracy; that is the ratio of correct
classifications to all classifications.

## Methods


## Results
| Method		| Parameters					| Preprocessing			| Features							| Accuracy		|
| ----			| ----							| ----					| ----								| ----			|
| k-NN			| k=3, p=2, uniform weights		| -						| 784x1 all pixels, single channel	| 0.8527		|

## Usage