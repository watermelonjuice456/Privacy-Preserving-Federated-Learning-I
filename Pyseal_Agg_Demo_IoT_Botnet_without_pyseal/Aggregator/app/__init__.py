import json
import logging
import os

import requests

from flask import Flask, jsonify, send_file, request
from flask_cors import CORS, cross_origin

import numpy as np

from app.constant.http.error import SERVER_OK, SERVER_OK_MESSAGE

from config.flask_config import UPDATE_MODEL_ENDPOINT, DefaultConfig

aggregation_weightage = {1:0.2978, 2:0.3580, 3:0.3442}

def create_app(config_object=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    CORS(app)
    weights_clients = {1:None, 2:None, 3:None}
    
    if config_object is None:
        # load the instance config, if it exists, when not testing
        app.config.from_object(DefaultConfig)
    else:
        # load the test config if passed in
        app.config.from_object(config_object)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    logging.info("Setting up Aggregator!")

    server_ip = app.config.get("SERVER_IP")
     
    # a simple page that says hello
    @app.route('/')
    def hello():
        return "Hello from Aggregator!"

    #client send post request to this API to save the weights
    @app.route('/save_weights', methods=['POST'])
    def save_weights():
        #ask worker to send weights with worker_id
        content = request.json
        weights_json = content['weights']
        weights = [np.asarray(i) for i in weights_json]
        worker_id = content['worker_id']
        weights_clients[worker_id] = weights
        #implement a save weight function
        return jsonify({
            'success': True,
            'error_code': SERVER_OK,
            'error_message': SERVER_OK_MESSAGE,
        })

    #no other component calls this function, API endpoint not important!
    @app.route('/agg_val')
    def model_aggregation():
        #implement aggregation function here
        '''
        weight, num_party = seal.aggregate_encrypted_weights()
        '''

        if not weights_clients[1] or not weights_clients[2] or not weights_clients[3]:
            return jsonify({
                'success': True,
                'error_code': "Some client doesn't provide the weights",
                'error_message': SERVER_OK_MESSAGE,
            })
        aggregated_weight = aggregation_weightage[1]*np.array(weights_clients[1])+aggregation_weightage[2]*np.array(weights_clients[2])+aggregation_weightage[3]*np.array(weights_clients[3])

        res = {
            "weights": [i.tolist() for i in aggregated_weight]
        }

        response = requests.post(server_ip + UPDATE_MODEL_ENDPOINT, json=res)

        res["update_status_code"] = response.status_code
        return jsonify({
            'success': True,
            'error_code': SERVER_OK,
            'error_message': SERVER_OK_MESSAGE,
            # 'result': res,
        })

    return app
