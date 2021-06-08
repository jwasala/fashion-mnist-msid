import numpy as np

from utils import mnist_reader
from models import knn
from models import preprocessing

X_train, y_train = mnist_reader.load_mnist('data', kind='train')
X_test, y_test = mnist_reader.load_mnist('data', kind='t10k')

X_train = preprocessing.resize(X_train, 16)
X_test = preprocessing.resize(X_test, 16)

k = np.arange(1, 20)
best_error, best_k, errors = knn.model_selection_knn(X_test, X_train, y_test, y_train, k)

print('best error:', 1 - best_error)
print('best k:', best_k)
