import numpy as np


def euclidean_distance(X, X_train):
    """
    Zwróć odległość euklidesową dla obiektów ze zbioru *X* od obiektów z *X_train*.

    :param X: zbiór porównywanych obiektów N1xD
    :param X_train: zbiór obiektów do których porównujemy N2xD
    :return: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    """
    dist = np.zeros((X.shape[0], X_train.shape[0]))

    for i in range(X.shape[0]):
        for j in range(X_train.shape[0]):
            dist[i][j] = np.linalg.norm(X[i] - X_train[j])

    return dist


def sort_train_labels_knn(Dist, y):
    """
    Posortuj etykiety klas danych treningowych *y* względem prawdopodobieństw
    zawartych w macierzy *Dist*.

    :param Dist: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    :param y: wektor etykiet o długości N2
    :return: macierz etykiet klas posortowana względem wartości podobieństw
        odpowiadającego wiersza macierzy Dist N1xN2

    Do sortowania użyj algorytmu mergesort.
    """

    return y[Dist.argsort(kind='mergesort')]


def p_y_x_knn(y, k):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) każdej z klas dla obiektów
    ze zbioru testowego wykorzystując klasyfikator KNN wyuczony na danych
    treningowych.

    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najbliższych sasiadow dla KNN
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" N1xM
    """
    return [[l / k for l in np.bincount([y[i][j] for j in range(k)], minlength=np.max(y[0]) + 1)] for i in range(np.shape(y)[0])]


def classification_error(p_y_x, y_true):
    """
    Wyznacz błąd klasyfikacji.

    :param p_y_x: macierz przewidywanych prawdopodobieństw - każdy wiersz
        macierzy reprezentuje rozkład p(y|x) NxM
    :param y_true: zbiór rzeczywistych etykiet klas 1xN
    :return: błąd klasyfikacji
    """
    (n, m) = np.shape(p_y_x)
    return np.sum([1 if (m - np.argmax(np.flip(p_y_x[i])) - 1) != y_true[i] else 0 for i in range(n)]) / n


def model_selection_knn(X_val, X_train, y_val, y_train, k_values):
    """
    Wylicz bład dla różnych wartości *k*. Dokonaj selekcji modelu KNN
    wyznaczając najlepszą wartość *k*, tj. taką, dla której wartość błędu jest
    najniższa.

    :param X_val: zbiór danych walidacyjnych N1xD
    :param X_train: zbiór danych treningowych N2xD
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartości parametru k, które mają zostać sprawdzone
    :return: krotka (best_error, best_k, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_k" to "k" dla którego błąd był
        najniższy, a "errors" - lista wartości błędów dla kolejnych
        "k" z "k_values"
    """
    dist = euclidean_distance(X_val, X_train)
    sorted_train_labels = sort_train_labels_knn(dist, y_train)
    p_y_x = [p_y_x_knn(sorted_train_labels, k) for k in k_values]
    errors = [classification_error(i, y_val) for i in p_y_x]

    lowest_err_index = np.argmin(errors)
    return errors[lowest_err_index], k_values[lowest_err_index], errors
