import numpy as np
import skimage.transform


def resize(X, n):
    """
    Resize MNIST data from 28x28 to nxn pixels.
    :param X: Mx784 matrix
    :return: Mxn matrix
    """
    X_copy = np.zeros((np.shape(X)[0], n * n))

    for i in range(len(X)):
        X_copy[i] = skimage.transform.resize(X[i], (n * n,))

    return X_copy
