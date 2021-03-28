import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from os import path
from datetime import datetime
import csv
from random import shuffle
import pickle



# Define a simple sequential model
def create_model():
    input_dim = 115
    model = tf.keras.Sequential()

    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.Dense(int(0.75 * input_dim), activation='relu'))
    model.add(layers.Dense(int(0.5 * input_dim), activation='relu'))
    model.add(layers.Dense(int(0.33 * input_dim), activation='relu'))
    model.add(layers.Dense(int(0.25 * input_dim), activation='relu'))
    model.add(layers.Dense(int(0.33 * input_dim), activation='relu'))
    model.add(layers.Dense(int(0.5 * input_dim), activation='relu'))
    model.add(layers.Dense(int(0.75 * input_dim), activation='relu'))
    model.add(layers.Dense(input_dim))
    Optimizer = keras.optimizers.SGD()
    model.compile(optimizer=Optimizer, loss="mean_squared_error", metrics=['accuracy'])
    model.load_weights('initial_weight/weight.h5')
    logging.info("model initialized!")
    return model

#do mirai and BASHLITE together
def evaluation(model):
    validation_data = pd.read_csv('../../train_test_validation/validation_centralized.csv')
    validation_x = validation_data.iloc[:, :-2]
    #model = keras.models.load_model(model_path)

    X_predict = model.predict(validation_x)
    #print(X_predict)

    mse = np.mean(np.power(validation_x - X_predict, 2), axis=1)
    #print(mse)

    # calculate threshold
    tr = mse.mean() + mse.std()
    #print('threshold = ' + str(tr))

    devices_BASHLITE = ['Ecobee_Thermostat', 'Provision_PT_737E_Security_Camera', 'Philips_B120N10_Baby_Monitor',
                        'Provision_PT_838_Security_Camera', 'SimpleHome_XCS7_1002_WHT_Security_Camera',
                        'Danmini_Doorbell',
                        'SimpleHome_XCS7_1003_WHT_Security_Camera',
                        'Samsung_SNH_1011_N_Webcam', 'Ennio_Doorbell']
    test_data_BASHLITE = pd.read_csv('../../train_test_validation/test_BASHLITE.csv')

    devices_mirai = ['Ecobee_Thermostat', 'Provision_PT_737E_Security_Camera', 'Philips_B120N10_Baby_Monitor',
                     'Provision_PT_838_Security_Camera', 'SimpleHome_XCS7_1002_WHT_Security_Camera',
                     'Danmini_Doorbell',
                     'SimpleHome_XCS7_1003_WHT_Security_Camera']
    test_data_mirai = pd.read_csv('../../train_test_validation/test_mirai.csv')

    result_BASHLITE = []
    for device in devices_BASHLITE:
        test_data_single_device = test_data_BASHLITE[test_data_BASHLITE['device'] == device]
        test_predict = model.predict(test_data_single_device.iloc[:, :-2])

        mse_test = np.mean(np.power(test_data_single_device.iloc[:, :-2] - test_predict, 2), axis=1)

        predictions = (mse_test > tr).astype(int)
        tn, fp, fn, tp = confusion_matrix(test_data_single_device['label'], predictions,
                                          labels=[0, 1]).ravel()
        accuracy = (tp + tn) / (tn + fp + fn + tp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        temp = [device, tn, fp, fn, tp, accuracy, precision, recall]
        result_BASHLITE.append(temp)

    with open('logbook_accuray_BASHLIE.csv', 'a', newline='') as logbook:
        writer = csv.writer(logbook)
        writer.writerow(['[' + str(datetime.now()) + ']' + 'testing result of BASHLITE'])
        for i in range(len(result_BASHLITE)):
            writer.writerow(result_BASHLITE[i])
    logbook.close()


    result_mirai = []
    for device in devices_mirai:
        test_data_single_device = test_data_mirai[test_data_mirai['device'] == device]
        test_predict = model.predict(test_data_single_device.iloc[:, :-2])

        mse_test = np.mean(np.power(test_data_single_device.iloc[:, :-2] - test_predict, 2), axis=1)

        predictions = (mse_test > tr).astype(int)

        tn, fp, fn, tp = confusion_matrix(test_data_single_device['label'], predictions,
                                          labels=[0, 1]).ravel()
        accuracy = (tp + tn) / (tn + fp + fn + tp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        temp = [device, tn, fp, fn, tp, accuracy, precision, recall]
        result_mirai.append(temp)
    with open('logbook_accuray_mirai.csv', 'a', newline='') as logbook:
        writer = csv.writer(logbook)
        writer.writerow(['[' + str(datetime.now()) + ']' + 'testing result of mirai'])
        for i in range(len(result_mirai)):
            writer.writerow(result_mirai[i])
    logbook.close()

    return result_BASHLITE, result_mirai


'''
    
    '''


