import ast
import datetime
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

from config.flask_config import MODEL_SAVE_FILE, WORKER1_IP,WORKER2_IP, WORKER3_IP, AGGREGATOR_IP,APP_ROOT
from model import create_model, evaluation
from tensorflow import keras
import numpy as np

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'launcher.sqlite'),
    )
    
    CORS(app)
    
    worker1_ip = WORKER1_IP
    worker2_ip = WORKER2_IP
    worker3_ip = WORKER3_IP
    aggregator_ip = AGGREGATOR_IP

    model_save_path = os.path.join("output", MODEL_SAVE_FILE)
       
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
        return 'Server running!'

    @app.route('/update_model_weights_enc', methods=['POST'])
    def update_model_weights():
        content = request.json
        weights_json = content['weights']
        weights = [numpy.asarray(i) for i in weights_json]
        model.set_weights(weights)
        evaluation(model)
        return jsonify({
            'success': True,
            'error_code': SERVER_OK,
            'error_message': SERVER_OK_MESSAGE,
        })

    @app.route('/get_model')
    def get_model():
        return send_file(os.path.join(APP_ROOT, model_save_path))

    #clients call this API to get weights of aggregated model
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



    @app.route('/model_evaluation')
    def model_evaluation():
        evaluation(model)
        return ('Model evaluation done and result saved to local files!')

    return app
