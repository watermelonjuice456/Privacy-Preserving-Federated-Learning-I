# PySEAL Aggregation Demo server

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

Alternatively, you can just run docker-compose to build this project: 
```bash
docker-compose up 
```

Currently, the server is set up to run at port `7000`.

## APIs 

The current APIs available in this server: 
*   GET `/get_params`
    Used to get the parameters for the PySEAL in json format.
    Example input: 
    ```
    localhost:7000/get_params 
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
    
    
*   GET `/localhost:7000/get_saved_params`
    Used to get the parameters in a binary file that can be loaded into PySEAL. 
    
    Example input: 
    ```
    localhost:7000/get_saved_params 
    ```
        

*   GET `/`
    Used to test whether server is alive. It will return `Hello, World!` if invoked. 

*   GET `/get_model`
    Used to get the base model in `.h5` format. 
    Example input: 
    ```
    localhost:7000/get_model 
    ``` 

*   GET `/get_model_weights`
    Used to get the model weights in json format. 
    Example input: 
    ```
    localhost:7000/model_weights 
    ``` 
    
    Example output: 
    ```json
    {
        "error_code": 0,
        "error_message": "",
        "result": {
            "weights": [weights_1,..., weights_n]
        },
        "success": true
    }
    ```

*   POST `update_model_weights_enc`
    Used to update the model based on the encrypted weights. 
    Example input: 
    ```
    localhost:7000/update_model_weights_enc 
    ``` 
    Request input format: 
    ```json
    {
        "weights": [weights_1,..., weights_n],
        "num_party": n
    }
    ```
    Where 
    *   `weights` denotes the weights of each layer as a list of weights. 
    *   `num_party` denotes the number of workers that are participating in the aggregated value. 

*   GET `/get_public_key`  
    Used to get the public key of the server.  
    It will returns a binary file which contains the public key used by the server. 
    
    Example input: 
    ```
    localhost:7000/get_public_key 
    ``` 

*   GET `/evaluate_model`
    Used to check the current accuracy of the model stored in the server. 
    This is obtained by running the model through the specified test set. 
    
    Example input: 
    ```
    localhost:7000/evaluate_model 
    ```
    
    Example output: 
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

*   POST `/predict_model`
    Used to get the current model stored in the server to predict the given image. 
    It will return the best prediction as well as its confidence level. 
    It will also return the confidence levels for other classes as well. 
    
    Example input:
    ```
    localhost:7000/predict_model 
    ```
    
    Example output: 
    ```json
    {
       "error_code":0,
       "error_message":"",
       "result":{
          "best_prediction":{
             "confidence":0.7917090058326721,
             "res":2
          },
          "predictions":[
             4.0364333016359297e-20,
             0.00017330220725852996,
             0.7917090058326721,
             2.237759067242296e-29,
             4.978668085436899e-12,
             2.7264536802189366e-18,
             1.8633147240730346e-16,
             0.20811770856380463,
             6.393519805292413e-26,
             6.77957245898142e-09
          ]
       },
       "success":true
    }
    ```