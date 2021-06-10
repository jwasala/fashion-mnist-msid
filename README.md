## Introduction
This repository presents a machine learning solution for classifying Fashion MNIST dataset.

Fashion MNIST dataset contains 70,000 examples of 28x28 grayscale images of garment, each labelled with one of the ten
categories. Examples have been split into two groups, 1) test set with 10,000 elements, 2) training set with 60,000
elements.

A target algorithm would predict labels of images in test set, after training with labeled training set. Quality of an
algorithm can be measured in terms of its accuracy; that is, the ratio of correct classifications to the number
of all classifications.

## Methods
### Models
I considered two models for completing this task, 1) *k*-nearest neighbors classifier, 2) ...

#### *k*-nearest neighbors
The *k*-nearest neighbors (k-NN) is a non-parametric algorithm that labels new objects based on labels of *k* most
similar training objects.

As a measure of similarity, depending on features, I used
[Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) or
[Hamming distance](https://en.wikipedia.org/wiki/Hamming_distance). Euclidean distance is the length of a line segment
between two points in Euclidean space and thus allows for object features to be any  real numbers. If features are
binary, Hamming distance can be used.	

### Feature extraction
I tried several techniques of feature extraction, 1) thresholding, 2) ...
and no extraction at all—in some tests, I passed all image data (all pixels' brightnesses) as a vector of features.

#### Thresholding
In the thresholding process, every pixel's brightness is replaced with 0, if its brightness (represented by an
integer from 0 to 255) is lower than some constant T, or 1 otherwise.

Motivation for using thresholding in the feature extraction process is that the category of a piece of clothing is
primarily designated by its shape rather than its color. Thresholding removes information about specific shade of
presented garment, effectively leaving only information about its shape.

Following figures present a subset of 72 samples from MNIST dataset before and after applying thresholding T=10.

![Before applying thresholding](thresholding_before.png "Before applying thresholding")
![After applying thresholding](thresholding_after.png "After applying thresholding")

## Results
| Method		| Parameters									| Preprocessing			| Features								| Accuracy		|
| ----			| ----											| ----					| ----									| ----			|
| **k-NN**		| **k=7, Hamming distance, uniform weights**	| **Thresholding T=10**	| **784x1 [0..1] binary pixels**		| **0.8634**	|
| k-NN			| k=7, Hamming distance, uniform weights		| Thresholding T=13		| 784x1 [0..1] binary pixels			| 0.8608		|
| k-NN			| k=7, Hamming distance, uniform weights		| Thresholding T=4		| 784x1 [0..1] binary pixels			| 0.8572		|
| k-NN			| k=3, Euclidean distance, uniform weights		| -						| 784x1 [0..225] pixels, single channel	| 0.8527		|

## Usage
In order to reproduce the results, download or clone this repository and install requirements with `py -m pip install
-r requirements.txt` (or its equivalent on your machine) from within project's root directory. Then refer to `py
main.py -h`
for further information.

Python 3.9 is recommended. Repository already contains Fashion MNIST dataset, so no additional files have to be put in
the project directory.