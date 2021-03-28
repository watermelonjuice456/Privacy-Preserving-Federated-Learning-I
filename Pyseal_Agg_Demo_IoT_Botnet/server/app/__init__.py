import json
import logging
import os

import requests

import numpy
from PIL import Image

import base64

from flask import Flask, jsonify, send_file, request
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

from app.constant.http.error import SERVER_OK, SERVER_OK_MESSAGE
from app.utils.python_seal import PySeal

from config.flask_config import PARAMS_SAVE_FILE, MODEL_SAVE_FILE, CIPHERTEXT_SAVE_FILE, PUBLIC_KEY_SAVE_FILE, APP_ROOT, AGGREGATE_VALUE, TRAIN_LOCAL_MODEL, WORKER1_IP,WORKER2_IP, WORKER3_IP, AGGREGATOR_IP
from model import create_model, evaluation
from tensorflow import keras
import numpy as np

UPLOAD_MNIST = './MNIST-images'

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'launcher.sqlite'),
    )
    
    CORS(app)
    app.config['UPLOAD_FOLDER'] = UPLOAD_MNIST
    
    worker1_ip = WORKER1_IP
    worker2_ip = WORKER2_IP
    worker3_ip = WORKER3_IP
    aggregator_ip = AGGREGATOR_IP 
       
    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Setup PySeal
    parms_save_path = os.path.join("output", PARAMS_SAVE_FILE)
    model_save_path = os.path.join("output", MODEL_SAVE_FILE)
    logging.warning(f"Path is {parms_save_path}")
    pubkey_save_path = os.path.join("config", PUBLIC_KEY_SAVE_FILE)
    cipher_save_path = os.path.join("output", CIPHERTEXT_SAVE_FILE)
    seal = PySeal(parms_save_path, pubkey_save_path, cipher_save_path)

    # Setup model

    #model = create_model()
    # Load model
    if os.path.exists(model_save_path):
    #if os.path.exists(MODEL_SAVE_FILE):
        try:
            model = keras.models.load_model(model_save_path)
            #model = keras.models.load_model(MODEL_SAVE_FILE)
            
        except:
            logging.warning("Error loading previous model! creating a new one...")
    else:
        logging.info("Previous model not found! creating a new one...")
        model = create_model()
    #model_save_path = os.path.join("output", MODEL_SAVE_FILE)
    model.save(model_save_path)
    logging.info("Demo model saved!")

    # a simple page that says hello
    @app.route('/')
    def hello():
        return 'Hello from server!'

    @app.route('/get_params')
    def get_params():
        res = seal.get_param_info()
        return jsonify({
            'success': True,
            'error_code': SERVER_OK,
            'error_message': SERVER_OK_MESSAGE,
            'result': res
        })

    @app.route('/get_saved_params')
    def get_saved_params():
        return send_file(parms_save_path)

    @app.route('/get_public_key')
    def get_public_key():
        return send_file(os.path.join(APP_ROOT, pubkey_save_path))


    @app.route('/get_model')
    def get_model():
        return send_file(os.path.join(APP_ROOT, model_save_path))

    @app.route('/get_model_weights')
    def get_model_weights():
        res = {
            "weights": [i.tolist() for i in model.get_weights()]
        }
        return jsonify({
            'success': True,
            'error_code': SERVER_OK,
            'error_message': SERVER_OK_MESSAGE,
            'result': res
        })

    @app.route('/update_model_weights_enc', methods=['POST', 'GET'])
    def update_model_weights():
        content = request.json
        weights = content['weights']
        num_party = content['num_party']
        logging.info("Num workers involved = {}".format(num_party))
        update_weights = seal.decode_and_decrypt_weight_layers(weights, num_party)

        for idx, weight in enumerate(model.get_weights()):
            shape = weight.shape
            new_weight = update_weights[idx]
            new_weight = numpy.resize(new_weight, shape)
            update_weights[idx] = new_weight
            logging.debug("layer weight {} = {}".format(idx, update_weights[idx]))

        model.set_weights(update_weights)
        model_save_path = os.path.join("output", MODEL_SAVE_FILE)
        model.save(model_save_path)
        print('model weight updated!')
        #evaluate_model()
        return jsonify({
            'success': True,
            'error_code': SERVER_OK,
            'error_message': SERVER_OK_MESSAGE,
        })
   

    @app.route("/evaluate_model", methods=["POST", "GET"])
    def evaluate_model():
        # file = request.files["image"]
        # img = Image.open(file.stream)
        
        #img = Image.open('./MNIST-images/mnist.png')

        evaluation(model)
       
        return jsonify({
            'success': True,
            'error_code': SERVER_OK,
            'error_message': SERVER_OK_MESSAGE
        })
    
    # http://localhost:7000/trainepochs?nEpochs=3
    @app.route("/trainepochs", methods=["GET"])
    def train_global_n_epochs():
    
        query_parameters = request.args
        n_epoch = query_parameters.get('nEpochs')
        logging.info(n_epoch)
        
        for i in range(int(n_epoch)):
		               
               res = {"nEpoch": i+1}
               logging.info('epoch started')   
               
               response_worker1 = requests.post("http://localhost:7103/train")
               logging.info(response_worker1.json())
               
               response_worker2 = requests.post(worker2_ip + TRAIN_LOCAL_MODEL)
               logging.info(response_worker2.json())
               
               response_worker3 = requests.post(worker3_ip + TRAIN_LOCAL_MODEL) 
               logging.info(response_worker3.json())
                      
               response_aggregator = requests.post(aggregator_ip + AGGREGATE_VALUE)
               logging.info('epoch completed')
                      
        return jsonify({
            'success': True,
            'num_epoch': n_epoch
        })

    return app
