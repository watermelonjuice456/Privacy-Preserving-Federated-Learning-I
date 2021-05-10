import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def load_models():
    arr = []
    models = glob.glob('client_models/*.npy')
    print(models)

    for i in models:
        arr.append(np.load(i, allow_pickle = True))
    return np.array(arr)


#get training dataset for all the clients
def load_sizes():
    sizes = []
    file_name_lists = glob.glob('client_datasize/*.txt')
    for file_name in file_name_lists:
        size = open(file_name, 'r').readline()
        sizes.append(int(size))
    return sizes

def fl_average():
    models = load_models()
    sizes = load_sizes()
    Sum = sum(sizes)
    models = load_models()
    models = np.array(models)
    sizes = load_sizes()
    Sum = sum(sizes)
    weights = np.zeros(models[0].shape)

    for i, size in enumerate(sizes):
        model = models[i]
        print('weights')
        weights = weights + model * size / Sum
    return weights

def build_model(average_weights):
    input_dim = 115
    model = tf.keras.Sequential()
    model.add(layers.Dense(int(0.75 * input_dim), activation='relu', input_shape=(115,)))
    model.add(layers.Dense(int(0.5 * input_dim), activation='relu'))
    model.add(layers.Dense(int(0.33 * input_dim), activation='relu'))
    model.add(layers.Dense(int(0.25 * input_dim), activation='relu'))
    model.add(layers.Dense(int(0.33 * input_dim), activation='relu'))
    model.add(layers.Dense(int(0.5 * input_dim), activation='relu'))
    model.add(layers.Dense(int(0.75 * input_dim), activation='relu'))
    model.add(layers.Dense(input_dim))

    Optimizer = keras.optimizers.SGD(learning_rate=0.012)
    model.compile(optimizer=Optimizer, loss="mean_squared_error", metrics=['accuracy'])
    model.set_weights(average_weights)
    model.compile(optimizer=Optimizer, loss="mean_squared_error", metrics=['accuracy'])

    return model


def save_local_model(model):
    model.save("aggregated_models/agg_model.h5")
    print('Local model update to local storage!')

def model_aggregation():

    average_weights = fl_average()

    model = build_model(average_weights)

    save_local_model(model)


