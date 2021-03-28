# PySEAL Aggregation Demo 

This project is based on the [baseline code](baseline/Aggregator.py) (under `baseline/Aggregator.py`).  

## Subproject division

This project is further divided to 3 different subproject to handle different party roles in the network. Namely: 
*   Server  
    Located under `server`. By default, the server will be hosted on port `7000`.
*   Workers  
    Located under `workers`. By default, it will spawn two workers with port `7101` and `7102`. 
*   Aggregator  
    Located under `Aggregator`. By default, the server will be hosted on port `7200`.

Please refer to the respective subprojects' readme for more information. 

## Setup Notes

All instances in this project requires PySeal library to be installed to be able to run properl. Since this library is unavailable in `pip`, 
it is decided that the PySeal library will be built from scratch and used as a base image for this server. 
As a consequence, the PySeal library must first be build and published locally in the docker repository hosting this API server. 
The current working implementation is available at the `seal-python` submodule. This submodule is currently available from [here](https://github.com/hanstananda/SEAL-Python)

## Usage guide 
1.  Start the `server`, followed by `workers` and `aggregator` respectively. 
2.  To start training, go to any worker endpoints and invoke the `/train` API.
    ```
    localhost:7101/train 
    ```
    The training is currently set to perform 1 training epoch. 
    Afterwards, it will automatically send the resulting encrypted weight to the Aggregator service. 
3.  After the workers have finished their training, you can invoke the `/agg_val` API on the Aggregator service 
    to aggregate the weights and transfer it to the server. 
    ```
    localhost:7200/agg_val 
    ```
4.  To check the current accuracy of the model stored in the server, we can use the `/evaluate_model` API. 
    ```
    localhost:7000/evaluate_model
    ```
    You will get a json response containing current test accuracy and loss value, which looks something like this: 
    ```json
    {
        "error_code": 0,
        "error_message": "",
        "result": {
            "accuracy": 0.9749000072479248,
            "loss": 0.08522004634141922
        },
        "success": true
    }
    ```
    

### Model Used
The model description is located under `server/model/` package. 
The current model is built based on the [Keras Simple MNIST Convnet](https://keras.io/examples/vision/mnist_convnet/) example, which is:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 56)        16184     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 56)          0         
_________________________________________________________________
flatten (Flatten)            (None, 1400)              0         
_________________________________________________________________
dropout (Dropout)            (None, 1400)              0         
_________________________________________________________________
dense (Dense)                (None, 10)                14010     
=================================================================
Total params: 30,514
Trainable params: 30,514
Non-trainable params: 0
_________________________________________________________________
``` 

### Performance benchmarks 
The following is the result of model shown above:  
#### Full dataset baseline (without secure aggregation)
*   After 1 epoch:
    ```
    Test loss: 0.09053578227758408
    Test accuracy: 0.972599983215332
    ```
*   After 5 epochs: 
    ```
    Test loss: 0.04196620732545853
    Test accuracy: 0.9855999946594238
    ```
*   After 10 epochs: 
    ```
    Test loss: 0.02865796536207199
    Test accuracy: 0.9907000064849854
    ```
*   After 15 epochs: 
    ```
    Test loss: 0.025408869609236717
    Test accuracy: 0.991100013256073
    ```
#### Secure aggregation with Full dataset for each worker
On the initial testing for training the secure aggregation, both workers are provided with the same full training dataset. The results are shown below:
*   After 1 epoch:
    ```
    Test loss: 0.08777600526809692
    Test accuracy: 0.9761999845504761
    ```
*   After 2 epochs:
    ```
    Test loss: 0.056792695075273514
    Test accuracy: 0.9824000000953674
    ```
*   After 3 epochs: 
    ```
    Test loss: 0.04806794226169586
    Test accuracy: 0.9843999743461609
    ```
*   After 4 epochs: 
    ```
    Test loss: 0.041380319744348526
    Test accuracy: 0.9868999719619751
    ```
*   After 5 epochs: 
    ```
    Test loss: 0.03970799595117569
    Test accuracy: 0.9876999855041504
    ```

1.  The encryption performed on the layer weights after training took 0.2075s (0.0093s) 

    Samples (in seconds): 
    ```
    0.21032500000000454
    0.20291400000000692
    0.20019999999999527 
    0.2029890000000023
    0.21514799999999923
    0.20154600000000755
    0.21778899999998202
    0.22577900000004547
    0.19980599999996684
    0.19823800000000347
    ```

2.  The aggregation performed on the layer weights from 2 workers took 0.1346s (0.0043s)

    Samples (in seconds): 
    ```
    0.138749
    0.13593299999999986
    0.12739699999999998
    0.13434999999999997
    0.136384999999999
    ```
3.  The decryption performed on the layer weights took 0.0894s (0.0036s)

    Samples (in seconds): 
    ```
    0.0958129999999997 
    0.08885700000000085
    0.08803299999999936
    0.08703599999999945
    0.08745500000000206
    ```

#### Secure aggregation with full dataset distributed evenly for each worker 
We then started to test with 2 workers having 30000 datasets each, totalling 60000(size of MNIST Train dataset)
*   After 1 epoch:
    ```
    INFO:root:Test loss: 0.13688255846500397
    INFO:root:Test accuracy: 0.9606000185012817
    ```
*   After 2 epochs: 
    ```
    INFO:root:Test loss: 0.08176544308662415
    INFO:root:Test accuracy: 0.977400004863739
    ```
*   After 3 epochs: 
    ```
    INFO:root:Test loss: 0.06307029724121094
    INFO:root:Test accuracy: 0.9804999828338623
    ```
*   After 4 epochs: 
    ```
    INFO:root:Test loss: 0.05522022768855095
    INFO:root:Test accuracy: 0.983299970626831
    ```
*   After 5 epochs: 
    ```
    INFO:root:Test loss: 0.05013105273246765
    INFO:root:Test accuracy: 0.9850999712944031
    ```
*   After 6 epochs: 
    ```
    INFO:root:Test loss: 0.045372750610113144
    INFO:root:Test accuracy: 0.9854000210762024
    ```
*   After 7 epochs:
    ```
    INFO:root:Test loss: 0.0451381579041481
    INFO:root:Test accuracy: 0.9861999750137329
    ```

*   After 8 epochs:
    ```
    INFO:root:Test loss: 0.04052318260073662
    INFO:root:Test accuracy: 0.9868000149726868
    ```
    
*   After 9 epochs:
    ```
    INFO:root:Test loss: 0.03903285786509514
    INFO:root:Test accuracy: 0.9873999953269958
    ```
    
*   After 10 epochs:
    ```
    INFO:root:Test loss: 0.04045134037733078
    INFO:root:Test accuracy: 0.9861999750137329
    ```    

### Additional Notes
*   After some trials, it is known that the maximum poly modulus degree supported is around 32768(2^15). 
    Therefore, the maximum number of parameters(matrix size) in layer that can be encrypted is 16384(2^14). 
    To address this issue, we added the support for splitting & merging mechanism for the layer in case if the number 
    of parameters is bigger than specified. 

 