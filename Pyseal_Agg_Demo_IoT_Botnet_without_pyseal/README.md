## Source code for Federated Learning without secure aggregation
Another version of Federated Learning that runs on Flask 
This project is based on the [Pyseal_Agg_Demo-Master](https://github.com/wangyingwwyy/Privacy-Preserving-Federated-Learning-I/tree/master/PySEAL_Agg_Demo-master/PySEAL_Agg_Demo-master)

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

### Usage guide 
1.  Start the `server`, followed by `workers`, `aggregator` and  `epoch_control` respectively. 
2.  Use the following URL with user-defined epoch_number to start the training.
    ```
    localhost:7104/trainepochs?nepochs=epoch_number
    ```

3.  To check the current accuracy of the model stored in the server, we can use the `/model_evaluation` API. 
    ```
    localhost:7000/model_evaluation
    ```
    The evaluation results will be write to csv files under server file
