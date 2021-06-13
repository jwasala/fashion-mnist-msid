from tensorflow.keras import layers, models
from constants import NUMBER_OF_CLASSES

preprocessing = models.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(28, 28, 1)),
    layers.experimental.preprocessing.RandomTranslation(0, 0.15),
    layers.experimental.preprocessing.Rescaling(1. / 255),
])

cnn_1 = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.Conv2D(128, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(NUMBER_OF_CLASSES, activation='softmax')
])

cnn_2 = models.Sequential([
    layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    layers.Conv2D(64, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUMBER_OF_CLASSES, activation='softmax')
])