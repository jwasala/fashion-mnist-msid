def thresholding(X, t):
    X_copy = X.copy()

    for i in range(len(X_copy)):
        for j in range(len(X_copy[i])):
            X_copy[i][j] = 1 if X_copy[i][j] >= t else 0

    return X_copy
