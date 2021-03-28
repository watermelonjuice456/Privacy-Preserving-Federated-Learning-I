import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from os import path
from datetime import datetime
import csv
from random import shuffle
import logging

class ModelNN:
    def __init__(self, h5_file, model_save_path, worker_id):
        # Setup model to be trained
        self.worder_id = worker_id
        with open(model_save_path, 'wb+') as f:
            f.write(h5_file)

        # This is our model definition taken from server
        self.model = tf.keras.models.load_model(model_save_path)


    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()
   
    def train(self, train_data, validation_data, worker_id):
       
        batch_sizes = {1:302, 2:363, 3:349}
        batch_size = batch_sizes[worker_id]



        self.model.fit(train_data.iloc[:, :-2], train_data.iloc[:, :-2],
                            batch_size=batch_size,
                            epochs=1,
                            validation_data=(validation_data.iloc[:, :-2], validation_data.iloc[:, :-2]),)

        logging.info("Finished training for 1 epoch!")
