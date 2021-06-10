import numpy as np

from utils import mnist_reader, feature_extraction
from models import knn

X_train, y_train = mnist_reader.load_mnist('data', kind='train')
X_test, y_test = mnist_reader.load_mnist('data', kind='t10k')

X_train = X_train
X_test = X_test
y_train = y_train
y_test = y_test

X_train = feature_extraction.thresholding(X_train, 12)
X_test = feature_extraction.thresholding(X_test, 12)

k = np.arange(1, 20)
best_error, best_k, errors = knn.model_selection_knn(X_test, X_train, y_test, y_train, k, knn.hamming_distance)

print('best error:', 1 - best_error)
print('best k:', best_k)
