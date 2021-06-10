from constants import NUMBER_OF_CLASSES
import numpy as np


def thresholding(X, t):
    X_copy = X.copy()

    for i in range(len(X_copy)):
        for j in range(len(X_copy[i])):
            X_copy[i][j] = 1 if X_copy[i][j] >= t else 0

    return X_copy


def hamming_distance(X, X_train):
    return ((1 - X) @ X_train.T) + (X @ (1 - X_train).T)


def euclidean_distance(X, X_train):
    dist = np.zeros((X.shape[0], X_train.shape[0]))

    for i in range(X.shape[0]):
        dist[i] = np.linalg.norm(X[i] - X_train, axis=1)

    return dist


def sort_train_labels_knn(Dist, y):
    return y[Dist.argsort(kind='mergesort')]


def p_y_x_knn(y, k):
    return [[l / k for l in np.bincount([y[i][j] for j in range(k)], minlength=NUMBER_OF_CLASSES)] for i in range(np.shape(y)[0])]


def classification_error(p_y_x, y_true):
    (n, m) = np.shape(p_y_x)
    return np.sum([1 if (m - np.argmax(np.flip(p_y_x[i])) - 1) != y_true[i] else 0 for i in range(n)]) / n


def model_selection_knn(X_val, X_train, y_val, y_train, k_values, dist_function):
    dist = dist_function(X_val, X_train)
    sorted_train_labels = sort_train_labels_knn(dist, y_train)
    p_y_x = []

    for k in k_values:
        p_y_x.append(p_y_x_knn(sorted_train_labels, k))

    errors = [classification_error(i, y_val) for i in p_y_x]

    lowest_err_index = np.argmin(errors)
    return errors[lowest_err_index], k_values[lowest_err_index], errors
