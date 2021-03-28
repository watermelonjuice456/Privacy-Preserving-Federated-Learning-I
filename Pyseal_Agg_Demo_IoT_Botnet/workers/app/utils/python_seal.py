import base64
import logging
import os
import time

import numpy
from seal import *


class PySeal:
    scale: float
    parms: EncryptionParameters

    # Setup PySeal
    def __init__(self, seal_server_params, pubkey_file, pubkey_save_path, cipher_save_path):

        self.parms = EncryptionParameters(scheme_type.CKKS)
        self.cipher_save_path = cipher_save_path

        # Set Coeff modulus
        coeff_modulus = [SmallModulus(i) for i in seal_server_params["coeff_modulus"]]
        self.parms.set_coeff_modulus(coeff_modulus)

        # Set Plain modulus
        self.parms.set_plain_modulus(seal_server_params["plain_modulus"])

        # Set Poly modulus degree
        self.parms.set_poly_modulus_degree(seal_server_params["poly_modulus_degree"])

        self.max_encrypted_size = self.parms.poly_modulus_degree() // 2 - 1

        context = SEALContext.Create(self.parms)
        self.print_parameters(context)

        self.scale = pow(2.0, 40)

        with open(pubkey_save_path, 'wb+') as f:
            f.write(pubkey_file)
            logging.warning("Pubkey saved!")

        public_key = PublicKey()
        public_key.load(context, pubkey_save_path)

        self.encoder = CKKSEncoder(context)

        self.encryptor = Encryptor(context, public_key)

    def get_param_info(self):
        res = {
            "scheme": self.get_scheme_name(self.parms.scheme()),
            "poly_modulus_degree": self.parms.poly_modulus_degree(),
            "coeff_modulus": [i.value() for i in self.parms.coeff_modulus()],
            "coeff_modulus_size": [i.bit_count() for i in self.parms.coeff_modulus()],
            "plain_modulus": self.parms.plain_modulus().value(),
        }
        return res

    def encode_and_encrypt(self, vector_1d):
        vector = DoubleVector(vector_1d)
        plain = Plaintext()  # Initialize the plaintexts
        self.encoder.encode(vector, self.scale, plain)  # Encoding
        encrypted = Ciphertext()  # Initialize the ciphertexts
        self.encryptor.encrypt(plain, encrypted)  # Encryption

        return encrypted

    def save_cipher_and_encode64(self, encrypted_object: Ciphertext):
        encrypted_object.save(self.cipher_save_path)
        with open(self.cipher_save_path, 'rb') as f:
            encrypted_text = f.read()
            # print(encrypted_text)
            res = base64.b64encode(encrypted_text)
        os.remove(self.cipher_save_path)
        # logging.debug("Resulting call type: {}".format(type(res)))
        # logging.debug(res[-10:])
        return res.decode('utf-8')

    def encrypt_layer_weights(self, layer_weights):
        weights = []
        start_time = time.perf_counter()

        for idx, layer_weight in enumerate(layer_weights):
            logging.debug("layer weight {} = {}".format(idx, layer_weight))
            logging.debug(layer_weight.shape)
            # Reshape to 1D vector
            vector_size = layer_weight.size
            reshaped_layer_weight = numpy.resize(layer_weight, (vector_size//self.max_encrypted_size+1, min(self.max_encrypted_size, vector_size)))
            logging.debug(reshaped_layer_weight.shape)
            # Cast to python native list then pass to SEAL library to be encrypted
            weights.append(self.partition_and_encrypt_layer(reshaped_layer_weight))

        time_elapsed = time.perf_counter() - start_time
        logging.info(f"Time taken for encryption is {time_elapsed} s")
        return weights

    def partition_and_encrypt_layer(self, reshaped_layer_weight):
        encrypted_weights = []
        for layer_partition in reshaped_layer_weight:
            encrypted_object = self.encode_and_encrypt(layer_partition.tolist())
            encrypted_plaintext = self.save_cipher_and_encode64(encrypted_object)
            encrypted_weights.append(dict({
                "partition_size": len(layer_partition),
                "encrypted_content": encrypted_plaintext
            }))
        return encrypted_weights

    @staticmethod
    def get_scheme_name(scheme):
        if scheme == scheme_type.BFV:
            scheme_name = "BFV"
        elif scheme == scheme_type.CKKS:
            scheme_name = "CKKS"
        else:
            scheme_name = "unsupported scheme"
        return scheme_name

    @classmethod
    def print_parameters(cls, context):
        context_data = context.key_context_data()
        scheme_name = cls.get_scheme_name(context_data.parms().scheme())
        print("/")
        print("| Encryption parameters:")
        print("| scheme: " + scheme_name)
        print("| poly_modulus_degree: " +
              str(context_data.parms().poly_modulus_degree()))
        print("| coeff_modulus size: ", end="")
        coeff_modulus = context_data.parms().coeff_modulus()
        coeff_modulus_sum = 0
        for j in coeff_modulus:
            coeff_modulus_sum += j.bit_count()
        print(str(coeff_modulus_sum) + "(", end="")
        for i in range(len(coeff_modulus) - 1):
            print(str(coeff_modulus[i].bit_count()) + " + ", end="")
        print(str(coeff_modulus[-1].bit_count()) + ") bits")
        if context_data.parms().scheme() == scheme_type.BFV:
            print("| plain_modulus: " +
                  str(context_data.parms().plain_modulus().value()))
        print("\\")
