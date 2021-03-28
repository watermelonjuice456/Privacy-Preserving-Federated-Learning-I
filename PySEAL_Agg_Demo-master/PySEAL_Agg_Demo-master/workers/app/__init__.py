import json
import logging
import os
import time

import numpy
import requests

from app.model import ModelNN
from app.utils import python_seal
import tensorflow as tf
from tensorflow import keras

from flask import Flask, jsonify, send_file
from flask_cors import CORS, cross_origin

from app.constant.http.error import SERVER_OK, SERVER_OK_MESSAGE
from app.utils.dataset_loader import LoaderMNIST
from config.flask_config import PARAMS_JSON_ENDPOINT, SERVER_GET_PUBLIC_KEY_ENDPOINT, DefaultConfig, \
    SERVER_MODEL_ENDPOINT, SERVER_WEIGHT_ENDPOINT, SAVE_WEIGHT_MATRIX_ENDPOINT, MODEL_SAVE_FILE, PUBLIC_KEY_SAVE_FILE, \
    CIPHERTEXT_SAVE_FILE


def create_app(config_object=None, worker_id=1):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'launcher.sqlite'),
    )
    CORS(app)

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

    aggregator_ip = app.config.get("AGGREGATOR_IP")
    server_ip = app.config.get("SERVER_IP")

    # Setup PySeal
    logging.info("Setting up PySeal for Worker {}!".format(worker_id))

    server_params_text = requests.get(server_ip + PARAMS_JSON_ENDPOINT).text
    server_params_json = json.loads(server_params_text)
    server_params = server_params_json["result"]
    # Get public key from server
    pubkey_file = requests.get(server_ip + SERVER_GET_PUBLIC_KEY_ENDPOINT).content
    pubkey_save_path = os.path.join(app.instance_path, PUBLIC_KEY_SAVE_FILE)
    cipher_save_path = os.path.join(app.instance_path, CIPHERTEXT_SAVE_FILE)
    seal = python_seal.PySeal(server_params, pubkey_file, pubkey_save_path, cipher_save_path)

    logging.info("PySeal for Worker {} set up successfully!".format(worker_id))

    # Setup Model
    model_save_path = os.path.join(app.instance_path + MODEL_SAVE_FILE)
    h5_file = requests.get(server_ip + SERVER_MODEL_ENDPOINT).content
    model_nn = ModelNN(h5_file, model_save_path)
    dataset = LoaderMNIST(2)

    logging.info("Model for Worker {} initiated successfully!".format(worker_id))

    # a simple page that says hello
    @app.route('/')
    def hello():
        return "Hello from Worker {}!".format(worker_id)

    # To check whether worker params successfully set up
    @app.route('/get_params')
    def get_params():
        res = seal.get_param_info()
        return jsonify({
            'success': True,
            'error_code': SERVER_OK,
            'error_message': SERVER_OK_MESSAGE,
            'result': res
        })

    @app.route("/reload_weight")
    def reload_weight():
        weight_string = requests.get(server_ip + SERVER_WEIGHT_ENDPOINT).text
        weight_json = json.loads(weight_string)['result']
        weights = [numpy.asarray(i) for i in weight_json["weights"]]
        model_nn.set_weights(weights)

        return jsonify({
            'success': True,
            'error_code': SERVER_OK,
            'error_message': SERVER_OK_MESSAGE
        })

    @app.route("/train", methods = ['POST'])
    def train():
        reload_weight()
        x_train, y_train = dataset.get_train_data_partitions(worker_id)
        model_nn.train(x_train, y_train)
        request = {
            "weights": seal.encrypt_layer_weights(model_nn.get_weights())
        }
        # for i in request["weights"]:
        #     logging.info("Data type is {}".format(type(i)))
        requests.post(aggregator_ip + SAVE_WEIGHT_MATRIX_ENDPOINT, json=request)
               
        response = jsonify({
            'success': True,
            'error_code': SERVER_OK,
            'error_message': SERVER_OK_MESSAGE
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    return app
