import logging

import keras
import numpy
import tensorflow as tf
import os
import requests
from keras.models import Model


class ModelNN:
    def __init__(self, h5_file, model_save_path):
        # Setup model to be trained

        with open(model_save_path, 'wb+') as f:
            f.write(h5_file)

        # This is our model definition taken from server
        self.model = tf.keras.models.load_model(model_save_path)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def train(self, x_train, y_train):

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255

        # Make sure images have shape (28, 28, 1)
        x_train = numpy.expand_dims(x_train, -1)

        print("x_train shape:", x_train.shape)
        print(x_train.shape[0], "train samples")

        # convert class vectors to binary class matrices
        num_classes = 10
        y_train = keras.utils.to_categorical(y_train, num_classes)

        history = self.model.fit(
            x_train,
            y_train,
            batch_size=128,
            epochs=1,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            validation_split=0.1
        )
        logging.info("Finished training for 1 epoch!")
