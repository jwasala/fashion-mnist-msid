from utils import mnist_reader
from models import knn, cnn
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import models

if __name__ == '__main__':
    X_train, y_train = mnist_reader.load_mnist('data', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data', kind='t10k')

    if '-knn' in sys.argv:
        X_train = knn.thresholding(X_train, 12)
        X_test = knn.thresholding(X_test, 12)

        k = np.arange(1, 20)
        best_error, best_k, errors = knn.model_selection_knn(X_test, X_train, y_test, y_train, k, knn.hamming_distance)

        print('best error:', 1 - best_error)
        print('best k:', best_k)
    elif '-cnn' in sys.argv:
        X_train = tf.reshape(X_train, (-1, 28, 28, 1))
        X_test = tf.reshape(X_test, (-1, 28, 28, 1))

        model = models.Sequential([
            cnn.preprocessing,
            cnn.cnn_1 if '1' in sys.argv else cnn.cnn_2
        ])

        if '1' in sys.argv:
            cnn.cnn_1.summary()
        else:
            cnn.cnn_2.summary()

        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=60)

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

        print('loss', test_loss)
        print('accuracy', test_acc)
    else:
        print('Use -knn for presentation of k-nearest neighbors algorithm or -cnn (1|2) for presentation of '
              'convolutional neural network.')
