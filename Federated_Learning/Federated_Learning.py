import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
from os import path
from datetime import datetime
import csv


batch_size = {1:302, 2:363, 3:349}
aggregation_weight = {1:0.2978, 2:0.3580, 3:0.3442}

def read_train_validation_data():

    train_client1 = pd.read_csv('../train_validation_test_minmax/train_client1_FL.csv')
    train_client2 = pd.read_csv('../train_validation_test_minmax/train_client2_FL.csv')
    train_client3 = pd.read_csv('../train_validation_test_minmax/train_client3_FL.csv')

    validation_client1 = pd.read_csv('../train_validation_test_minmax/validation_client1_FL.csv')
    validation_client2 = pd.read_csv('../train_validation_test_minmax/validation_client2_FL.csv')
    validation_client3 = pd.read_csv('../train_validation_test_minmax/validation_client3_FL.csv')

    train_client1 = train_client1.sample(frac = 1)
    train_client2 = train_client2.sample(frac=1)
    train_client3 = train_client3.sample(frac=1)

    validation_client1 = validation_client1.sample(frac = 1)
    validation_client2 = validation_client2.sample(frac=1)
    validation_client3 = validation_client3.sample(frac=1)


    return train_client1, train_client2, train_client3, validation_client1, validation_client2, validation_client3

def build_model():
    input_dim = 115
    model = tf.keras.Sequential()
    model.add(layers.Input(shape = (input_dim,)))
    model.add(layers.Dense(int(0.75 * input_dim), activation='relu'))
    model.add(layers.Dense(int(0.5 * input_dim), activation='relu'))
    model.add(layers.Dense(int(0.33 * input_dim), activation='relu'))
    model.add(layers.Dense(int(0.25 * input_dim), activation='relu'))
    model.add(layers.Dense(int(0.33 * input_dim), activation='relu'))
    model.add(layers.Dense(int(0.5 * input_dim), activation='relu'))
    model.add(layers.Dense(int(0.75 * input_dim), activation='relu'))
    model.add(layers.Dense(input_dim))
    return model

def train_model():
    train_client1, train_client2, train_client3, validation_client1, validation_client2, validation_client3 = read_train_validation_data()
    train_data = {1:train_client1, 2:train_client2, 3:train_client3}
    validation_data = {1:validation_client1, 2:validation_client2, 3:validation_client3}

    Optimizer = keras.optimizers.SGD()
    model1 = build_model()
    model2 = build_model()
    model3 = build_model()

    model = build_model()

    models = {1:model1, 2:model2, 3:model3}

    model.compile(optimizer=Optimizer, loss="mean_squared_error", metrics=['accuracy'])

    dummy = pd.DataFrame()
    train_data_client = {1:dummy, 2:dummy, 3:dummy}
    average_weights = 0
    epoch = 0
    #os.mkdir('centralized_model')

    #around 40 epochs
    for i in range(1, 12801):
        for j in range(1, 4):
            if i > 1:
                models[j].set_weights(average_weights)
            else:
                models[j].compile(optimizer=Optimizer, loss="mean_squared_error", metrics=['accuracy'])
                models[j].load_weights('../initial_weight/weight.h5')

        if train_data[1].shape[0] < 302:
            print('finish training on one epoch')
            train_data[1] = pd.read_csv('../train_validation_test_minmax/train_client1.csv')
            train_data[2] = pd.read_csv('../train_validation_test_minmax/train_client2.csv')
            train_data[3] = pd.read_csv('../train_validation_test_minmax/train_client3.csv')
            model.save('global_model_minmax_local/model_{:04d}.h5'.format(epoch))
            epoch += 1

        for j in range(1, 4):
            print('training on local model ' + str(j))
            train_data[j] = train_data[j].sample(frac=1)
            train_data_client[j] = train_data[j][:batch_size[j]]
            print(train_data_client[j])
            print(train_data[j])
            train_data[j] = train_data[j].iloc[batch_size[j]:]
            print(train_data[j])
            models[j].fit(train_data_client[j].iloc[:, :-2], train_data_client[j].iloc[:, :-2],
                          batch_size=batch_size[j],
                          epochs=1,
                          validation_data=(validation_data[j].iloc[:, :-2], validation_data[j].iloc[:, :-2]))


        for j in range(1,4):
            models[j].fit(train_data_client[j].iloc[:, :-2], train_data_client[j].iloc[:, :-2],
                          batch_size=batch_size[j],
                          epochs=1,
                          validation_data=(validation_data[j].iloc[:, :-2], validation_data[j].iloc[:, :-2]))
        average_weights = aggregation_weight[1] * np.array(models[1].get_weights()) + aggregation_weight[2] * np.array(
            models[2].get_weights()) + aggregation_weight[3] * np.array(models[3].get_weights())
        print(average_weights[0])
        model.set_weights(average_weights)
        #model.save('global_model/model_{:04d}.h5'.format(i))


        #aggregation
        average_weights = aggregation_weight[1] * np.array(models[1].get_weights())+aggregation_weight[2] * np.array(models[2].get_weights()) + aggregation_weight[3] * np.array(models[3].get_weights())
        print(average_weights[0])




def evaluate_model(virus, Path, threshold):

    model = keras.models.load_model(Path)

    #tr = 0.019348176634629
    tr = threshold


    if not path.exists('logbook_global_model.csv'):
        with open('logbook_global_model.csv', 'w', newline='') as logbook:
            writer = csv.writer(logbook)
            writer.writerow(['Device Name', 'TN', 'FP', 'FN', 'TP', 'Accuracy', 'Precision', 'Recall'])
        logbook.close()

    if virus == 'BASHLITE':
        devices = ['Ecobee_Thermostat', 'Provision_PT_737E_Security_Camera', 'Philips_B120N10_Baby_Monitor',
                   'Provision_PT_838_Security_Camera', 'SimpleHome_XCS7_1002_WHT_Security_Camera',
                   'Danmini_Doorbell',
                   'SimpleHome_XCS7_1003_WHT_Security_Camera',
                   'Samsung_SNH_1011_N_Webcam', 'Ennio_Doorbell']
        test_data = pd.read_csv('../train_validation_test_minmax/test_BASHLITE.csv')


    elif virus == 'mirai':
        devices = ['Ecobee_Thermostat', 'Provision_PT_737E_Security_Camera', 'Philips_B120N10_Baby_Monitor',
                   'Provision_PT_838_Security_Camera', 'SimpleHome_XCS7_1002_WHT_Security_Camera',
                   'Danmini_Doorbell',
                   'SimpleHome_XCS7_1003_WHT_Security_Camera']
        test_data = pd.read_csv('../train_validation_test_minmax/test_mirai.csv')

    log = []
    for device in devices:
        test_data_single_device = test_data[test_data['device'] == device]
        print('test data for device' + device)
        mse_store['label'] = test_data_single_device['label']
        mse_store['device'] = test_data_single_device['device']
        print(test_data_single_device)
        test_predict = model.predict(test_data_single_device.iloc[:, :-2])

        mse_test = np.mean(np.power(test_data_single_device.iloc[:, :-2] - test_predict, 2), axis=1)
        predictions = (mse_test > tr).astype(int)
        print('predicion results: ')
        print(predictions)


        tn, fp, fn, tp = confusion_matrix(test_data_single_device['label'], predictions,
                                          labels=[0, 1]).ravel()

        accuracy = (tp + tn) / (tn + fp + fn + tp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        temp = [device, tn, fp, fn, tp, accuracy, precision, recall,tr]
        log.append(temp)
        test_result = pd.concat([test_result, mse_store])
        mse_store = pd.DataFrame()

    with open('logbook_global_model.csv', 'a', newline='') as logbook:
        writer = csv.writer(logbook)
        writer.writerow(['[' + str(datetime.now()) + ']' + 'testing result of ' + virus])
        for i in range(len(log)):
            writer.writerow(log[i])
    logbook.close()


train_model()