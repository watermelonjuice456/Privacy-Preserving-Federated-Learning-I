import logging
import os

import numpy
import tensorflow as tf
from PIL.Image import Image
from tensorflow import keras

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)


# Define a simple sequential model
def create_model():
    model = tf.keras.Sequential([
        keras.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10,
                              activation=tf.nn.softmax,
                              kernel_constraint=tf.keras.constraints.MinMaxNorm(
                                  min_value=0.0, max_value=1.0, rate=1.0, axis=0
                              )
                              ),
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    model.summary()

    return model


def predict(model, image: Image):
    # Change to grayscale
    image = image.convert('L')
    # Change to size supported by the MNIST
    image = image.resize((input_shape[0], input_shape[1]))
    image_array = numpy.array(image).reshape(input_shape)
    image_array = tf.expand_dims(image_array, 0)

    predictions = model.predict(image_array)
    logging.debug(predictions)
    return predictions
