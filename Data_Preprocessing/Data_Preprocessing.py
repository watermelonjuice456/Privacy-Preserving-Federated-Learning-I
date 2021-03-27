import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# BASHLITE test data - for each device: 1000 benign + 1000 malicious
# Mirai test data - for each device: 1000 benign + 1000 malicious
# The rest : will be further split into train and test
# The path here is for my own use.
# Raw data in 'Data' path mentioned in this code please download from: https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT

def generate_test_without_scaling():
    devices_mirai = ['Ecobee_Thermostat', 'Philips_B120N10_Baby_Monitor', 'Danmini_Doorbell',
                     'Provision_PT_737E_Security_Camera',
                     'Provision_PT_838_Security_Camera', 'SimpleHome_XCS7_1002_WHT_Security_Camera',
                     'SimpleHome_XCS7_1003_WHT_Security_Camera']
    devices_BASHLITE = ['Ecobee_Thermostat', 'Philips_B120N10_Baby_Monitor', 'Danmini_Doorbell',
                        'Provision_PT_737E_Security_Camera',
                        'Provision_PT_838_Security_Camera', 'SimpleHome_XCS7_1002_WHT_Security_Camera',
                        'SimpleHome_XCS7_1003_WHT_Security_Camera', 'Samsung_SNH_1011_N_Webcam', 'Ennio_Doorbell']

    test_mirai_malicious = pd.DataFrame()
    test_mirai_benign = pd.DataFrame()
    test_BASHLITE_malicious = pd.DataFrame()
    test_BASHLITE_benign = pd.DataFrame()
    train_benign = pd.DataFrame()

    for device in devices_mirai:
        print('get benign data from device ' + device)
        benign_data_device = pd.read_csv('../Data/' + device + '/benign_traffic.csv')
        benign_data_device['label'] = 0
        benign_data_device['device'] = device
        print('benign data')
        print(benign_data_device)
        benign_data_device = benign_data_device.sample(frac=1)
        test_mirai_benign_device = benign_data_device[:1000]  # possible to modify the number here
        print('benign data for testing')
        print(test_mirai_benign_device)
        train_benign_device = benign_data_device[1000:]
        print('benign data for training')
        print(train_benign_device)
        test_mirai_benign = pd.concat([test_mirai_benign, test_mirai_benign_device])
        train_benign = pd.concat([train_benign, train_benign_device])

    for device in devices_mirai:
        print('get malicious data from device ' + device)
        ack_data = pd.read_csv('../Data/' + device + '/mirai_attacks/ack.csv')
        scan_data = pd.read_csv('../Data/' + device + '/mirai_attacks/scan.csv')
        syn_data = pd.read_csv('../Data/' + device + '/mirai_attacks/syn.csv')
        udp_data = pd.read_csv('../Data/' + device + '/mirai_attacks/udp.csv')
        udpplain_data = pd.read_csv('../Data/' + device + '/mirai_attacks/udpplain.csv')
        mirai_malicious_device = pd.concat([ack_data, scan_data, syn_data, udp_data, udpplain_data])
        mirai_malicious_device = mirai_malicious_device.sample(frac=1)
        test_mirai_malicous_device = mirai_malicious_device[0:1000]
        test_mirai_malicous_device['label'] = 1
        test_mirai_malicous_device['device'] = device
        print(test_mirai_malicous_device)
        test_mirai_malicious = pd.concat([test_mirai_malicious, test_mirai_malicous_device])

    test_BASHLITE_benign = test_mirai_benign.copy(deep=True)

    test_mirai = pd.concat([test_mirai_malicious, test_mirai_benign])
    test_mirai.to_csv('../train_validation_test_minmax/test_mirai_without_scaling.csv', index=False)

    for device in ['Samsung_SNH_1011_N_Webcam', 'Ennio_Doorbell']:
        print('get benign data from device ' + device)
        benign_data_device = pd.read_csv('../Data/' + device + '/benign_traffic.csv')
        benign_data_device['label'] = 0
        benign_data_device['device'] = device
        print('benign data')
        print(benign_data_device)
        benign_data_device = benign_data_device.sample(frac=1)
        test_BASHLITE_benign_device = benign_data_device[:1000]  # possible to modify the number here
        print('benign data for testing')
        print(test_BASHLITE_benign_device)
        train_benign_device = benign_data_device[1000:]
        print('benign data for training')
        print(train_benign_device)
        test_BASHLITE_benign = pd.concat([test_BASHLITE_benign, test_BASHLITE_benign_device])
        train_benign = pd.concat([train_benign, train_benign_device])

    train_benign.to_csv(
        '../train_validation_test_minmax/original_benign_for_train_validation_without_scaling.csv', index=False)

    for device in devices_BASHLITE:
        print('get malicious data from device ' + device)
        ack_data = pd.read_csv('../Data/' + device + '/gafgyt_attacks/combo.csv')
        scan_data = pd.read_csv('../Data/' + device + '/gafgyt_attacks/scan.csv')
        syn_data = pd.read_csv('../Data/' + device + '/gafgyt_attacks/junk.csv')
        udp_data = pd.read_csv('../Data/' + device + '/gafgyt_attacks/udp.csv')
        udpplain_data = pd.read_csv('../Data/' + device + '/gafgyt_attacks/tcp.csv')
        BASHLITE_malicious_device = pd.concat([ack_data, scan_data, syn_data, udp_data, udpplain_data])
        BASHLITE_malicious_device = BASHLITE_malicious_device.sample(frac=1)
        test_BASHLITE_malicous_device = BASHLITE_malicious_device[0:1000]
        test_BASHLITE_malicous_device['label'] = 1
        test_BASHLITE_malicous_device['device'] = device
        print(test_BASHLITE_malicous_device)
        test_BASHLITE_malicious = pd.concat([test_BASHLITE_malicious, test_BASHLITE_malicous_device])

    test_BASHLITE = pd.concat([test_BASHLITE_malicious, test_BASHLITE_benign])
    test_BASHLITE.to_csv('../train_validation_test_minmax/test_BASHLITE_without_scaling.csv', index=False)

def generate_train_validation_without_scaling():
    device_client1 = ['Ecobee_Thermostat', 'Provision_PT_737E_Security_Camera']
    device_client2 = ['Ennio_Doorbell', 'SimpleHome_XCS7_1002_WHT_Security_Camera']
    device_client3 = ['SimpleHome_XCS7_1003_WHT_Security_Camera']

    train_data = pd.read_csv(
        '../train_validation_test_minmax/original_benign_for_train_validation_without_scaling.csv')

    train_client1 = pd.DataFrame()
    train_client2 = pd.DataFrame()
    train_client3 = pd.DataFrame()

    validation_client1 = pd.DataFrame()
    validation_client2 = pd.DataFrame()
    validation_client3 = pd.DataFrame()

    for device in device_client1:
        print('get data from device' + device)
        device_data = train_data.loc[train_data['device'] == device]
        device_data = device_data.sample(frac=1)
        device_data = device_data[:45000]
        print(device_data)
        device_train, device_validation = train_test_split(device_data, test_size=0.33, random_state=42)
        train_client1 = pd.concat([train_client1, device_train])
        validation_client1 = pd.concat([validation_client1, device_validation])

    for device in device_client2:
        device_data = train_data.loc[train_data['device'] == device]
        device_data = device_data.sample(frac=1)
        device_data = device_data[:45000]
        device_train, device_validation = train_test_split(device_data, test_size=0.33, random_state=42)
        train_client2 = pd.concat([train_client2, device_train])
        validation_client2 = pd.concat([validation_client2, device_validation])

    for device in device_client3:
        device_data = train_data.loc[train_data['device'] == device]
        device_data = device_data.sample(frac=1)
        device_data = device_data[:45000]
        device_train, device_validation = train_test_split(device_data, test_size=0.33, random_state=42)
        train_client3 = pd.concat([train_client3, device_train])
        validation_client3 = pd.concat([validation_client3, device_validation])

    baby_monitor = train_data.loc[train_data['device'] == 'Philips_B120N10_Baby_Monitor']
    print('data for baby monitor')
    print(baby_monitor)
    baby_monitor = baby_monitor.sample(frac=1)
    baby_monitor_client1 = baby_monitor[:45000]
    baby_monitor_client2 = baby_monitor[45000:90000]
    baby_monitor_client3 = baby_monitor[90000:135000]

    baby_monitor_client1_train, baby_monitor_client1_validaiton = train_test_split(baby_monitor_client1, test_size=0.33,
                                                                                   random_state=42)
    baby_monitor_client2_train, baby_monitor_client2_validaiton = train_test_split(baby_monitor_client2, test_size=0.33,
                                                                                   random_state=42)
    baby_monitor_client3_train, baby_monitor_client3_validaiton = train_test_split(baby_monitor_client3, test_size=0.33,
                                                                                   random_state=42)

    train_client1 = pd.concat([train_client1, baby_monitor_client1_train])
    validation_client1 = pd.concat([validation_client1, baby_monitor_client1_validaiton])
    train_client2 = pd.concat([train_client2, baby_monitor_client2_train])
    validation_client2 = pd.concat([validation_client2, baby_monitor_client2_validaiton])
    train_client3 = pd.concat([train_client3, baby_monitor_client3_train])
    validation_client3 = pd.concat([validation_client3, baby_monitor_client3_validaiton])

    security_camera = train_data.loc[train_data['device'] == 'Provision_PT_838_Security_Camera']
    print('data for Provision_PT_838_Security_Camera')
    print(security_camera)
    security_camera = security_camera.sample(frac=1)
    security_camera_client2 = security_camera[:45000]
    security_camera_client3 = security_camera[45000:90000]

    security_camera_client2_train, security_camera_client2_validaiton = train_test_split(security_camera_client2,
                                                                                         test_size=0.33,
                                                                                         random_state=42)
    security_camera_client3_train, security_camera_client3_validaiton = train_test_split(security_camera_client3,
                                                                                         test_size=0.33,
                                                                                         random_state=42)

    train_client2 = pd.concat([train_client2, security_camera_client2_train])
    validation_client2 = pd.concat([validation_client2, security_camera_client2_validaiton])
    train_client3 = pd.concat([train_client3, security_camera_client3_train])
    validation_client3 = pd.concat([validation_client3, security_camera_client3_validaiton])

    doorbell = train_data.loc[train_data['device'] == 'Danmini_Doorbell']
    print('data for Danmini_Doorbell')
    print(doorbell)
    doorbell = doorbell.sample(frac=1)
    doorbell_client1 = doorbell[:21000]
    doorbell_client3 = doorbell[21000:]

    doorbell_client1_train, doorbell_client1_validaiton = train_test_split(doorbell_client1, test_size=0.33,
                                                                           random_state=42)
    doorbell_client3_train, doorbell_client3_validaiton = train_test_split(doorbell_client3, test_size=0.33,
                                                                           random_state=42)

    train_client1 = pd.concat([train_client1, doorbell_client1_train])
    validation_client1 = pd.concat([validation_client1, doorbell_client1_validaiton])
    train_client3 = pd.concat([train_client3, doorbell_client3_train])
    validation_client3 = pd.concat([validation_client3, doorbell_client3_validaiton])

    samsung = train_data.loc[train_data['device'] == 'Samsung_SNH_1011_N_Webcam']
    print('data for Samsung_SNH_1011_N_Webcam')
    print(samsung)
    samsung = samsung.sample(frac=1)
    samsung_client1 = samsung[:21000]
    samsung_client3 = samsung[21000:]

    samsung_client1_train, samsung_client1_validaiton = train_test_split(samsung_client1, test_size=0.33,
                                                                         random_state=42)
    samsung_client3_train, samsung_client3_validaiton = train_test_split(samsung_client3, test_size=0.33,
                                                                         random_state=42)

    train_client1 = pd.concat([train_client1, samsung_client1_train])
    validation_client1 = pd.concat([validation_client1, samsung_client1_validaiton])
    train_client3 = pd.concat([train_client3, samsung_client3_train])
    validation_client3 = pd.concat([validation_client3, samsung_client3_validaiton])

    centralized_train = pd.concat([train_client1, train_client2, train_client3])
    centralized_validation = pd.concat([validation_client1, validation_client2, validation_client3])

    train_client1.to_csv('../train_validation_test_minmax/train_client1_before_scaling.csv', index=False)
    validation_client1.to_csv('../train_validation_test_minmax/validation_client1_before_scaling.csv',
                              index=False)

    train_client2.to_csv('../train_validation_test_minmax/train_client2_before_scaling.csv', index=False)
    validation_client2.to_csv('../train_validation_test_minmax/validation_client2_before_scaling.csv',
                              index=False)

    train_client3.to_csv('../train_validation_test_minmax/train_client3_before_scaling.csv', index=False)
    validation_client3.to_csv('../train_validation_test_minmax/validation_client3_before_scaling.csv',
                              index=False)

    centralized_train.to_csv('../train_validation_test_minmax/centralized_train_before_scaling.csv', index=False)
    centralized_validation.to_csv('../train_validation_test_minmax/centralized_validation_before_scaling.csv',
                                  index=False)

def train_validation_scaling():
    train_centralized = pd.read_csv('../train_validation_test_minmax/centralized_train_before_scaling.csv')
    validation_centralized = pd.read_csv(
        '../train_validation_test_minmax/centralized_validation_before_scaling.csv')

    train_client1 = pd.read_csv('../train_validation_test_minmax/train_client1_before_scaling.csv')
    validation_client1 = pd.read_csv('../train_validation_test_minmax/validation_client1_before_scaling.csv')
    train_client2 = pd.read_csv('../train_validation_test_minmax/train_client2_before_scaling.csv')
    validation_client2 = pd.read_csv('../train_validation_test_minmax/validation_client2_before_scaling.csv')
    train_client3 = pd.read_csv('../train_validation_test_minmax/train_client3_before_scaling.csv')
    validation_client3 = pd.read_csv('../train_validation_test_minmax/validation_client3_before_scaling.csv')

    train_client1_FL = train_client1.copy(deep=True)
    validation_client1_FL = validation_client1.copy(deep=True)

    train_client2_FL = train_client2.copy(deep=True)
    validation_client2_FL = validation_client2.copy(deep=True)

    train_client3_FL = train_client3.copy(deep=True)
    validation_client3_FL = validation_client3.copy(deep=True)

    test_mirai = pd.read_csv('../train_validation_test_minmax/test_mirai_without_scaling.csv')
    test_mirai_client1 = test_mirai.copy(deep=True)
    test_mirai_client2 = test_mirai.copy(deep=True)
    test_mirai_client3 = test_mirai.copy(deep=True)

    test_BASHLITE = pd.read_csv('../train_validation_test_minmax/BASHLITE_1K_without_UDPTCP_before_scaling.csv')
    test_BASHLITE_client1 = test_BASHLITE.copy(deep=True)
    test_BASHLITE_client2 = test_BASHLITE.copy(deep=True)
    test_BASHLITE_client3 = test_BASHLITE.copy(deep=True)

    scaler_centralized = MinMaxScaler()
    scaler_client1 = MinMaxScaler()
    scaler_client2 = MinMaxScaler()
    scaler_client3 = MinMaxScaler()

    train_centralized[train_centralized.columns[:-2]] = scaler_centralized.fit_transform(
        train_centralized[train_centralized.columns[:-2]])
    validation_centralized[validation_centralized.columns[:-2]] = scaler_centralized.transform(
        validation_centralized[validation_centralized.columns[:-2]])

    train_client1_FL[train_client1_FL.columns[:-2]] = scaler_centralized.transform(
        train_client1_FL[train_client1_FL.columns[:-2]])
    validation_client1_FL[validation_client1_FL.columns[:-2]] = scaler_centralized.transform(
        validation_client1_FL[validation_client1_FL.columns[:-2]])

    train_client2_FL[train_client2_FL.columns[:-2]] = scaler_centralized.transform(
        train_client2_FL[train_client2_FL.columns[:-2]])
    validation_client2_FL[validation_client2_FL.columns[:-2]] = scaler_centralized.transform(
        validation_client2_FL[validation_client2_FL.columns[:-2]])

    train_client3_FL[train_client3_FL.columns[:-2]] = scaler_centralized.transform(
        train_client3_FL[train_client3_FL.columns[:-2]])
    validation_client3_FL[validation_client3_FL.columns[:-2]] = scaler_centralized.transform(
        validation_client3_FL[validation_client3_FL.columns[:-2]])

    test_mirai[test_mirai.columns[:-2]] = scaler_centralized.transform(test_mirai[test_mirai.columns[:-2]])
    test_BASHLITE[test_BASHLITE.columns[:-2]] = scaler_centralized.transform(test_BASHLITE[test_BASHLITE.columns[:-2]])

    print('saving centralized and FL files')
    train_centralized.to_csv('../train_validation_test_minmax/train_centralized.csv', index=False)
    validation_centralized.to_csv('../train_validation_test_minmax/validation_centralized.csv', index=False)
    train_client1_FL.to_csv('../train_validation_test_minmax/train_client1_FL.csv', index=False)
    validation_client1_FL.to_csv('../train_validation_test_minmax/validation_client1_FL.csv', index=False)
    train_client2_FL.to_csv('../train_validation_test_minmax/train_client2_FL.csv', index=False)
    validation_client2_FL.to_csv('../train_validation_test_minmax/validation_client2_FL.csv', index=False)
    train_client3_FL.to_csv('../train_validation_test_minmax/train_client3_FL.csv', index=False)
    validation_client3_FL.to_csv('../train_validation_test_minmax/validation_client3_FL.csv', index=False)
    test_mirai.to_csv('../train_validation_test_minmax/test_mirai.csv', index=False)
    test_BASHLITE.to_csv('../train_validation_test_minmax/test_BASHLITE.csv', index=False)

    print('scaling files for client1')
    train_client1[train_client1.columns[:-2]] = scaler_client1.fit_transform(train_client1[train_client1.columns[:-2]])
    validation_client1[validation_client1.columns[:-2]] = scaler_client1.transform(
        validation_client1[validation_client1.columns[:-2]])
    test_mirai_client1[test_mirai_client1.columns[:-2]] = scaler_client1.transform(
        test_mirai_client1[test_mirai_client1.columns[:-2]])
    test_BASHLITE_client1[test_BASHLITE_client1.columns[:-2]] = scaler_client1.transform(
        test_BASHLITE_client1[test_BASHLITE_client1.columns[:-2]])

    print('saving files for client1')
    train_client1.to_csv('../train_validation_test_minmax/train_client1.csv', index=False)
    validation_client1.to_csv('../train_validation_test_minmax/validation_client1.csv', index=False)
    test_mirai_client1.to_csv('../train_validation_test_minmax/test_mirai_client1.csv', index=False)
    test_BASHLITE_client1.to_csv('../train_validation_test_minmax/test_BASHLITE_client1.csv', index=False)

    print('scaling files for client2')
    train_client2[train_client2.columns[:-2]] = scaler_client2.fit_transform(train_client2[train_client2.columns[:-2]])
    validation_client2[validation_client2.columns[:-2]] = scaler_client2.transform(
        validation_client2[validation_client2.columns[:-2]])
    test_mirai_client2[test_mirai_client2.columns[:-2]] = scaler_client2.transform(
        test_mirai_client2[test_mirai_client2.columns[:-2]])
    test_BASHLITE_client2[test_BASHLITE_client2.columns[:-2]] = scaler_client2.transform(
        test_BASHLITE_client2[test_BASHLITE_client2.columns[:-2]])

    print('saving files for client2')
    train_client2.to_csv('../train_validation_test_minmax/train_client2.csv', index=False)
    validation_client2.to_csv('../train_validation_test_minmax/validation_client2.csv', index=False)
    test_mirai_client2.to_csv('../train_validation_test_minmax/test_mirai_client2.csv', index=False)
    test_BASHLITE_client2.to_csv('../train_validation_test_minmax/test_BASHLITE_client2.csv', index=False)

    print('scaling files for client3')
    train_client3[train_client3.columns[:-2]] = scaler_client3.fit_transform(train_client3[train_client3.columns[:-2]])
    validation_client3[validation_client3.columns[:-2]] = scaler_client3.transform(
        validation_client3[validation_client3.columns[:-2]])
    test_mirai_client3[test_mirai_client3.columns[:-2]] = scaler_client3.transform(
        test_mirai_client3[test_mirai_client3.columns[:-2]])
    test_BASHLITE_client3[test_BASHLITE_client3.columns[:-2]] = scaler_client3.transform(
        test_BASHLITE_client3[test_BASHLITE_client3.columns[:-2]])

    print('saving files for client3')
    train_client3.to_csv('../train_validation_test_minmax/train_client3.csv', index=False)
    validation_client3.to_csv('../train_validation_test_minmax/validation_client3.csv', index=False)
    test_mirai_client3.to_csv('../train_validation_test_minmax/test_mirai_client3.csv', index=False)
    test_BASHLITE_client3.to_csv('../train_validation_test_minmax/test_BASHLITE_client3.csv', index=False)

generate_test_without_scaling()
generate_train_validation_without_scaling()
train_validation_scaling()
