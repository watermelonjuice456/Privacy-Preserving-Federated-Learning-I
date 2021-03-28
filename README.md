# Privacy-Preserving-Federated-Learning-I
Source code of my Final Year Project - Privacy Preserving Federated Learning(I)

## Data Preprocessing
Download raw data at: https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT  
Run Data_Preprocessing.py to generate all the data files needed for this project

## Centralized Learning
In Centralized_Training.py: 
1. Run train_model(epochs) to run the model
2. Run threshold_calculation(Path) to get the threshold
3. Run evaluate_model() to get the TP, FP, TN, FN, accuracy, precision and recall of the model

## Individual Learning
In Individual_Training_Client1.py, Individual_Training_Client2.py, Individual_Training_Client3.py: 
1. Run train_model(epochs) to run the model
2. Run threshold_calculation(Path) to get the threshold
3. Run evaluate_model() to get the TP, FP, TN, FN, accuracy, precision and recall of the model

## Federated Learning
In Federated_Learning.py:
1. Run train_model(epochs) to run the model
2. Run evaluate_model() to get the TP, FP, TN, FN, accuracy, precision and recall of the model

## Pyseal_Aggregation_Demo_IoT_Botnet(Secure Aggregation Platform)
This project is based on the Pyseal_Agg_Demo-Mater

### Subproject division

This project is further divided to 4 different subproject to handle different party roles in the network. Namely: 
*   Server  
    Located under `server`. By default, the server will be hosted on port `7000`.
*   Workers  
    Located under `workers`. By default, it will spawn two workers with port `7101`,  `7102`, `7103`. 
*   Aggregator  
    Located under `Aggregator`. By default, the server will be hosted on port `7200`.
*   Epoch_Control
    Located under `Epoch Control`. By defaul, it will be hosted on port `7104`
    
Please refer to the respective subprojects' readme for more information. 

### Setup Notes

All instances in this project requires PySeal library to be installed to be able to run properl. Since this library is unavailable in `pip`, 
it is decided that the PySeal library will be built from scratch and used as a base image for this server. 
As a consequence, the PySeal library must first be build and published locally in the docker repository hosting this API server. 
The current working implementation is available at the `seal-python` submodule. This submodule is currently available from [here](https://github.com/hanstananda/SEAL-Python)

### Usage guide 
1.  Start the `server`, followed by `workers`, `aggregator` and  `epoch_control` respectively. 
2.  Use the following URL with user-defined epoch_number to start the training.
    ```
    localhost:7104/trainepochs?nepochs=epoch_number
    ```

3.  To check the current accuracy of the model stored in the server, we can use the `/evaluate_model` API. 
    ```
    localhost:7000/evaluate_model
    ```
    The evaluation results will be write to csv files under server file
 
## To do (for author's self reference only)
1. Change the path to make the code more user-friendly
2. Optimize the Data Preprocessing code and Invidiual Training code
3. Extend the functionality of Secure Aggregation Plaform
