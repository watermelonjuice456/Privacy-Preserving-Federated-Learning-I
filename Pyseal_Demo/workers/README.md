# PySEAL Workers service

## Prerequisites
1. Python 3 (Tested on Python 3.7)
2. PySEAL library

## Running the project

1.  Install the library requirements from pip
    ```bash
    pip install -r requirements.txt
    ```

2.  Run the python script
    ```
    python run_server.py
    ```
    
Currently, there will be two workers set up at port `7101` and `7102` respectively.

## APIs 

The current APIs available in this server: 
*   GET `/get_params`
    Used to get the parameters for the PySEAL in json format. 
    This is added to enable checking of PySEAL, so we can make sure that it is set up properly. 
    Example input: 
    ```
    localhost:7101/get_params 
    ``` 
    
    Example output: 
    ```json
    {
        "error_code": 0,
        "error_message": "",
        "result": {
            "coeff_modulus": [
                1152921504606748673,
                1099510890497,
                1099511480321,
                1152921504606830593
            ],
            "coeff_modulus_size": [
                60,
                40,
                40,
                60
            ],
            "plain_modulus": 0,
            "poly_modulus_degree": 8192,
            "scheme": "CKKS"
        },
        "success": true
    }
    ```

*   GET `/train`
    Used to train the model and send the encrypted weights after training to the aggregator service. 
    Currently set to only train for 1 epoch using MNIST dataset. 
    
    Example input: 
    ```
    localhost:7101/train 
    ``` 
    
    Example output: 
    ```json
    {
        "error_code": 0,
        "error_message": "",
        "success": true
    }
    ```

*   GET `/reload_weight`
    Used to reload the weight of the model in the worker based on the current weights in the server. 
    Used for synchronization / debugging purposes. 
    The underlying function will be called during each time `/train` API is called as well.
    
    Example input: 
    ```
    localhost:7101/reload_weight 
    ``` 
    
     Example output: 
    ```json
    {
        "error_code": 0,
        "error_message": "",
        "success": true
    }
    ```
    