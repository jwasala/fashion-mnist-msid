## Introduction
This repository presents a machine learning solution for classifying Fashion MNIST dataset.

Fashion MNIST dataset contains 70,000 examples of 28x28 grayscale images of garment, each labelled with one of the ten
categories. Examples have been split into two groups, 1) test set with 10,000 elements, 2) training set with 60,000
elements.

A target algorithm would predict labels of images in test set, after training with labeled training set. Quality of an
algorithm can be measured in terms of its accuracy; that is, the ratio of correct
classifications to the number of all classifications.

## Methods
I considered two methods for completing this task, 1) *k*-nearest neighbors classifier, 2) ...

### *k*-nearest neighbors
The *k*-nearest neighbors (k-NN) is a non-parametric algorithm that labels new objects based on labels of *k* most
similar training objects. As a measure of similarity, I used Euclidean distance, which is the length of a line segment
between two points in Euclidean space. This measure allows for any number of object features, and allows for object
features to be any real numbers.

Object features are represented by a vector of numbers. I tried several techniques of feature extraction, 1) ... 2) ...
and no extraction at all—in one of the tests, I passed all image data (all pixels' brightnesses) as a vector of
features.


## Results
| Method		| Parameters					| Preprocessing			| Features							| Accuracy		|
| ----			| ----							| ----					| ----								| ----			|
| k-NN			| k=3, p=2, uniform weights		| -						| 784x1 all pixels, single channel	| 0.8527		|

## Usage
In order to reproduce the results, download or clone this repository and install requirements with `py -m pip install
-r requirements.txt` (or its equivalent on your machine) from within project's root directory. Then refer to `py
main.py -h`
for further information.

Python 3.9 is recommended. Repository already contains Fashion MNIST dataset, so no additional files have to be put in
the project directory.