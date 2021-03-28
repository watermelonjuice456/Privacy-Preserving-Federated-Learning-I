# PySEAL Aggregator service

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
    
Currently, the Aggregator is set up to run at port `7200`.

## APIs 

The current APIs available in this Aggregator service: 
*   GET `/get_params`
    Used to get the parameters for the PySEAL in json format. 
    This is added to enable checking of PySEAL, so we can make sure that it is set up properly. 
    Example input: 
    ```
    localhost:7200/get_params 
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
    
*   GET `/agg_val`
    Used to aggregate the encrypted weights received from the worker nodes and then send it to the server. 
    
    Example input: 
    ```
    localhost:7200/agg_val 
    ``` 
    
    Example output: 
    ```json
    {
        "error_code": 0,
        "error_message": "",
        "success": true
    }
    ```

*   POST `/save_weights`
    Used to save the encrypted weights received from the workers to the queue. 
    Note that by design this API should not be called manually. 
    
    Example input: 
    ```
    localhost:7200/save_weights 
    ``` 
    Request input format: 
    ```json
    {
        "weights": [weights_1,..., weights_n]
    }
    ```
    
    Example output: 
    ```json
    {
        "error_code": 0,
        "error_message": "",
        "success": true
    }
    ```