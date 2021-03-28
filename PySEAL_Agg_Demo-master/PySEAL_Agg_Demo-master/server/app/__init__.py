import json
import logging
import os

import numpy
from PIL import Image
import base64

from flask import Flask, jsonify, send_file, request
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

from app.constant.http.error import SERVER_OK, SERVER_OK_MESSAGE
from app.utils.python_seal import PySeal

from config.flask_config import PARAMS_SAVE_FILE, MODEL_SAVE_FILE, CIPHERTEXT_SAVE_FILE, PUBLIC_KEY_SAVE_FILE, APP_ROOT
from model import create_model, num_classes, predict
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
    logging.warning(f"Path is {parms_save_path}")
    pubkey_save_path = os.path.join("config", PUBLIC_KEY_SAVE_FILE)
    cipher_save_path = os.path.join("output", CIPHERTEXT_SAVE_FILE)
    seal = PySeal(parms_save_path, pubkey_save_path, cipher_save_path)

    # Setup model

    model = create_model()
    # Load model
    if os.path.exists(MODEL_SAVE_FILE):
        try:
            model = keras.models.load_model(MODEL_SAVE_FILE)
        except:
            logging.warning("Error loading previous model! creating a new one...")
    else:
        logging.info("Previous model not found! creating a new one...")
    model_save_path = os.path.join("output", MODEL_SAVE_FILE)
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

    @app.route('/update_model_weights_enc', methods=['POST'])
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
        evaluate_model()
        return jsonify({
            'success': True,
            'error_code': SERVER_OK,
            'error_message': SERVER_OK_MESSAGE,
        })

    @app.route('/evaluate_model')
    def evaluate_model():
        (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
        # Scale images to the [0, 1] range
        x_test = x_test.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        x_test = numpy.expand_dims(x_test, -1)
        logging.info("{} test samples".format(x_test.shape[0]))
        y_test = keras.utils.to_categorical(y_test, num_classes)
        score = model.evaluate(x_test, y_test, verbose=0)
        logging.info("Test loss: {}".format(score[0]))
        logging.info("Test accuracy: {}".format(score[1]))
        res = {
            "loss": score[0],
            "accuracy": score[1],
        }
        return jsonify({
            'success': True,
            'error_code': SERVER_OK,
            'error_message': SERVER_OK_MESSAGE,
            'result': res
        })
        
    @app.route('/uploaduserimage', methods = ['POST'])
    def upload_user_image():
        
        if os.path.exists('./MNIST-images') is False:
            os.makedirs('./MNIST-images')        
     
        if request.method == 'POST':
            f = request.files['file']
            filename = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            os.rename('./MNIST-images' + '/' + filename, './MNIST-images' + '/' + 'mnist.png')

        response = jsonify({
                "statusCode": 200,
                "status": "The image uploaded successfully!",
                "image": " ",
        })
        
        response.headers.add('Access-Control-Allow-Origin', '*')
        
        return response

    @app.route('/save_drawing', methods=['POST'])
    def save_drawing_image_file():

        if os.path.exists('./MNIST-images') is False:
            os.makedirs('./MNIST-images') 

        data = request.get_json()

        if data is None:
            response = jsonify({
                    "statusCode": 200,
                    "status": "No valid request body, json missing!",
            })    
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response

        else:

            if request.method == 'POST':
                img_data = data['imgBase64']      
                img_data = img_data[22:]
                with open("MNIST-images/mnist.png","wb") as fh:
                    fh.write(base64.decodebytes(img_data.encode()))
   
            response = jsonify({
                    "statusCode": 200,
                    "status": "The drawing saved successfully",
                    "image": " ",
            })
        
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response

    @app.route("/predict_image", methods=["POST"])
    def predict_image():
        # file = request.files["image"]
        # img = Image.open(file.stream)
        
        img = Image.open('./MNIST-images/mnist.png')

        predictions = predict(model, img)
        res = {
            "predictions": [float(i) for i in predictions[0]],
            "best_prediction": {
                "res": int(numpy.argmax(predictions)),
                "confidence": float(numpy.amax(predictions))
            }
        }
        return jsonify({
            'success': True,
            'error_code': SERVER_OK,
            'error_message': SERVER_OK_MESSAGE,
            'result': str(res['best_prediction']["res"])
        })

    return app
