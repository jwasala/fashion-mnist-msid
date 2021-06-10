from tensorflow.keras import layers, models

preprocessing = models.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(28, 28, 1)),
    layers.experimental.preprocessing.RandomTranslation(0, 0.15),
    layers.experimental.preprocessing.Rescaling(1. / 255),
])

cnn_1 = models.Sequential([
    layers.Conv2D(784, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

cnn_2 = models.Sequential([
    layers.Conv2D(784, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

cnn_3 = models.Sequential([
    layers.Conv2D(128, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(784, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])