import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import os
from os import path
from datetime import datetime
import csv


def build_model():
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
    return model

def train_model(epochs):
    train_data = pd.read_csv('../../train_validation_test_minmax/train_client3.csv')
    train_data = train_data.sample(frac = 1)
    print('reading train_data')
    print(train_data)

    validation_data = pd.read_csv('../../train_validation_test_minmax/validation_client3.csv')
    validation_data = validation_data.sample(frac=1)
    print('reading validation_data')
    print(validation_data)

    model = build_model()
    model.load_weights('../../initial_weight/weight.h5')
    print('initial weight')
    print(model.get_weights())

    Optimizer = keras.optimizers.SGD()
    model.compile(optimizer=Optimizer, loss="mean_squared_error", metrics=['accuracy'])

    print('data for training')
    print(train_data.iloc[:, :-2])
    print('data for validation')
    print(validation_data.iloc[:, :-2])
    #os.mkdir('client3_model')
    checkpoint = ModelCheckpoint(filepath='client3_model/model_{epoch:04d}.h5', period = 1)

    history = model.fit(train_data.iloc[:, :-2], train_data.iloc[:, :-2],
                            batch_size=349,
                            epochs=epochs,
                            validation_data=(validation_data.iloc[:, :-2], validation_data.iloc[:, :-2]),
                            callbacks=[checkpoint]
                        )

    np.save('history_epoch_{}_client3.npy'.format(epochs), history.history)


def threshold_calculation(Path):
    validation_data = pd.read_csv('../../train_validation_test_minmax/validation_client3.csv')
    validation_x = validation_data.iloc[:, :-2]

    model = keras.models.load_model(Path)

    X_predict = model.predict(validation_x)

    mse = np.mean(np.power(validation_x - X_predict, 2), axis=1)
    print('power of np')
    print(np.power(validation_x - X_predict, 2))

    # calculate threshold
    tr = mse.mean() + mse.std()
    print('threshold = ' + str(tr))


def evaluate_model(virus, Path, threshold):
    model = keras.models.load_model(Path)
    #tr = 0.01874627619088995
    tr = threshold

    if not path.exists('logbook_client3.csv'):
        with open('logbook_client3.csv', 'w', newline='') as logbook:
            writer = csv.writer(logbook)
            writer.writerow(['Device Name', 'TN', 'FP', 'FN', 'TP', 'Accuracy', 'Precision', 'Recall'])
        logbook.close()


    if virus == 'BASHLITE':
        devices = ['Ecobee_Thermostat', 'Provision_PT_737E_Security_Camera', 'Philips_B120N10_Baby_Monitor',
                   'Provision_PT_838_Security_Camera', 'SimpleHome_XCS7_1002_WHT_Security_Camera',
                   'Danmini_Doorbell',
                   'SimpleHome_XCS7_1003_WHT_Security_Camera',
                   'Samsung_SNH_1011_N_Webcam', 'Ennio_Doorbell']
        test_data = pd.read_csv('../../train_validation_test_minmax/test_BASHLITE_client3.csv')
    elif virus == 'mirai':
        devices = ['Ecobee_Thermostat', 'Provision_PT_737E_Security_Camera', 'Philips_B120N10_Baby_Monitor',
                   'Provision_PT_838_Security_Camera', 'SimpleHome_XCS7_1002_WHT_Security_Camera',
                   'Danmini_Doorbell',
                   'SimpleHome_XCS7_1003_WHT_Security_Camera']
        test_data = pd.read_csv('../../train_validation_test_minmax/test_mirai_client3.csv')

    test_data = test_data.sample(frac = 1)

    test_result = pd.DataFrame()
    mse_store = pd.DataFrame()
    log = []
    for device in devices:
        test_data_single_device = test_data[test_data['device'] == device]
        mse_store['label'] = test_data_single_device['label']
        mse_store['device'] = test_data_single_device['device']
        test_predict = model.predict(test_data_single_device.iloc[:, :-2])
        mse_test = np.mean(np.power(test_data_single_device.iloc[:, :-2] - test_predict, 2), axis=1)
        mse_store['mse'] = mse_test
        predictions = (mse_test > tr).astype(int)
        print('predicion results: ')
        print(predictions)


        tn, fp, fn, tp = confusion_matrix(test_data_single_device['label'], predictions,
                                          labels=[0, 1]).ravel()

        accuracy = (tp + tn) / (tn + fp + fn + tp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        temp = [device, tn, fp, fn, tp, accuracy, precision, recall,tr]
        test_result = pd.concat([test_result, mse_store])
        mse_store= pd.DataFrame()
        log.append(temp)

    with open('logbook_client3.csv', 'a', newline='') as logbook:
        writer = csv.writer(logbook)
        writer.writerow(['[' + str(datetime.now()) + ']' + 'testing result of ' + virus])
        for i in range(len(log)):
            writer.writerow(log[i])
    logbook.close()

train_model(100)
