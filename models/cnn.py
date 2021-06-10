from tensorflow.keras import layers, models

preprocessing = models.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(28, 28, 1)),
    layers.experimental.preprocessing.RandomTranslation(0, 0.15),
    layers.experimental.preprocessing.Rescaling(1. / 255),
])

cnn_1 = models.Sequential([

])

cnn_2 = models.Sequential([
])

cnn_3 = models.Sequential([
])
