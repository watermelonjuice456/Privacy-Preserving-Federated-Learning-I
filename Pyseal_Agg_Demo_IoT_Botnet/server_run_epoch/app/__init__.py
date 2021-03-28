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

MODEL_SAVE_FILE = "demo_model.h5"
PARAMS_SAVE_FILE = "params.txt"
CIPHERTEXT_SAVE_FILE = "cipher.txt"
PUBLIC_KEY_SAVE_FILE = "public_key.pub"

AGGREGATE_VALUE = '/agg_val'

TRAIN_LOCAL_MODEL = '/train'
WORKER1_IP = 'http://localhost:7101'
WORKER2_IP = 'http://localhost:7102'
WORKER3_IP = 'http://localhost:7103'
AGGREGATOR_IP = "http://localhost:7200"


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    
    @app.route("/")
    def hello():
        return 'server for epoch running!'

    # http://localhost:7104/trainepochs?nEpochs=1
    @app.route("/trainepochs", methods=["GET"])
    def train_global_n_epochs():
    
        query_parameters = request.args
        n_epoch = query_parameters.get('nEpochs')
        logging.info(n_epoch)
        
        rounds = n_epoch *319
        
        for i in range(1,int(rounds)+1):
		               
               res = {"nEpoch": i+1}
               logging.info('epoch started')   
               
               response_worker1 = requests.post("http://localhost:7101/train")
               logging.info(response_worker1.json())
               print('worker 1 done!')
               
               response_worker2 = requests.post(WORKER2_IP + TRAIN_LOCAL_MODEL)
               logging.info(response_worker2.json())
               
               response_worker3 = requests.post(WORKER3_IP + TRAIN_LOCAL_MODEL) 
               logging.info(response_worker3.json())
               
               #http://localhost:7200/agg_val
               response_aggregator = requests.post(AGGREGATOR_IP + AGGREGATE_VALUE)
               logging.info('round completed')
               if int(rounds)%319 == 0:
                      logging.info('epoch completed')
                      
        return jsonify({
            'success': True,
            'num_epoch': n_epoch
        })    
    return app
