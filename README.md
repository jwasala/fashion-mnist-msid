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
I considered two models for completing this task, 1) *k*-nearest neighbors classifier, 2) convolutional neural network.

#### *k*-nearest neighbors
The *k*-nearest neighbors (k-NN) is a non-parametric algorithm that labels new objects based on labels of *k* most
similar training objects.

As a measure of similarity, depending on features, I used
[Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) or
[Hamming distance](https://en.wikipedia.org/wiki/Hamming_distance). Euclidean distance is the length of a line segment
between two points in Euclidean space and thus allows for object features to be any  real numbers. If all features are
binary, Hamming distance can be used.

#### Convolutional neural network
I tested several convolutional neural networks.

### Feature extraction
I tried several techniques of feature extraction, 1) thresholding, 2) ...
and no extraction at all—in some tests, I passed all image data (all pixels' brightnesses) as a vector of features.

#### Thresholding
In the thresholding process, every pixel's brightness is replaced with 0, if its brightness (represented by an
integer from 0 to 255) is lower than some constant T, or 1 otherwise.

Motivation for using thresholding in the feature extraction process is that the category of a piece of clothing is
primarily designated by its shape rather than its color. Thresholding removes information about specific shade of
presented garment, effectively leaving only information about its shape.

Following figure present a subset of 72 samples from MNIST dataset before and after applying thresholding T=10.

![Before applying thresholding](thresholding_before.png "Before applying thresholding")
![After applying thresholding](thresholding_after.png "After applying thresholding")

## Results
| Method		| Parameters									| Preprocessing									| Features												| Accuracy
| ----			| ----											| ----											| ----													| ----
| CNN (2)		| 2 Conv + Pooling, 1 903 082 params			| Normalization, random translation and flip	| 784x1 [0..225] pixels, single channel	| 0.9044		|
| CNN (2)		| 2 Conv + Pooling, 1 903 082 params			| -												| 784x1 [0..225] pixels, single channel	| 0.8892		|
| CNN (1)		| 3 Conv + Pooling, 533 994 params				| -												| 784x1 [0..225] pixels, single channel	| 0.8870		|
| k-NN			| k=7, Hamming distance, uniform weights		| Thresholding T=10								| 784x1 [0..1] binary pixels			| 0.8634		|
| k-NN			| k=7, Hamming distance, uniform weights		| Thresholding T=13								| 784x1 [0..1] binary pixels			| 0.8608		|
| k-NN			| k=7, Hamming distance, uniform weights		| Thresholding T=4								| 784x1 [0..1] binary pixels			| 0.8572		|
| k-NN			| k=3, Euclidean distance, uniform weights		| -												| 784x1 [0..225] pixels, single channel	| 0.8527		|

## Usage
In order to reproduce the results, download or clone this repository and install requirements with `py -m pip install
-r requirements.txt` (or its equivalent on your machine) from within project's root directory. Then refer to `py
main.py -h` for further information.

Python 3.9 is recommended. Repository already contains Fashion MNIST dataset, so no additional files have to be put in
the project directory.

## Appendix A. Architectures of used convolutional neural networks
### CNN no 1, 533,994 parameters, 3 layers and pooling
Layer type					| Output Shape		| Param
----						| ----				| ----
Conv2D						| 26, 26, 784		| 7840  
MaxPooling2D				| 13, 13, 784		| 0
Conv2D						| 11, 11, 64		| 451648
MaxPooling2D				| 5, 5, 64			| 0
Conv2D						| 3, 3, 64			| 36928
Flatten						| 576				| 0
Dense						| 64				| 36928
Dense						| 10				| 650

### CNN no 2, 1,903,082 parameters, 2 layers and pooling
Layer type					| Output Shape		| Param
----						| ----				| ----
Conv2D						| 26, 26, 784		| 7840
MaxPooling2D				| 13, 13, 784		| 0
Conv2D						| 11, 11, 128		| 903296
Flatten						| 15488				| 0
Dense						| 64				| 991296
Dense						| 10				| 650       
