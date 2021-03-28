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
    def __init__(self, parms_save_path, pubkey_save_path, cipher_save_path, poly_modulus_degree=8192 * 4):

        self.parms = EncryptionParameters(scheme_type.CKKS)
        self.cipher_save_path = cipher_save_path

        self.parms = EncryptionParameters(scheme_type.CKKS)

        EncryptionParameters.save(self.parms, parms_save_path)

        self.parms.set_poly_modulus_degree(poly_modulus_degree)
        self.parms.set_coeff_modulus(CoeffModulus.Create(
            poly_modulus_degree, [60, 40, 40, 60]))

        self.context = SEALContext.Create(self.parms)
        self.print_parameters(self.context)

        keygen = KeyGenerator(self.context)
        public_key = keygen.public_key()
        secret_key = keygen.secret_key()
        relin_keys = keygen.relin_keys()

        # Save public key file
        public_key.save(pubkey_save_path)

        self.decryptor = Decryptor(self.context, secret_key)

        self.encoder = CKKSEncoder(self.context)
        slot_count = self.encoder.slot_count()

    def get_param_info(self):
        res = {
            "scheme": self.get_scheme_name(self.parms.scheme()),
            "poly_modulus_degree": self.parms.poly_modulus_degree(),
            "coeff_modulus": [i.value() for i in self.parms.coeff_modulus()],
            "coeff_modulus_size": [i.bit_count() for i in self.parms.coeff_modulus()],
            "plain_modulus": self.parms.plain_modulus().value(),
        }
        return res

    def decode64_and_get_cipher_object(self, ins: str):
        # logging.debug(ins.encode('utf-8')[-10:])
        decoded = base64.b64decode(ins.encode('utf-8'))
        with open(self.cipher_save_path, 'wb') as f:
            f.write(decoded)
            cipher_object = Ciphertext()
            cipher_object.load(self.context, self.cipher_save_path)
        os.remove(self.cipher_save_path)
        return cipher_object

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

    def decode_and_decrypt_weight_layers(self, weights, num_party):
        update_weights = []
        for idx, layer in enumerate(weights):
            layer_weight = []
            for part_idx, partition_weight in enumerate(layer):
                encrypted_result = self.decode64_and_get_cipher_object(partition_weight["encrypted_content"])
                plain_result = Plaintext()
                self.decryptor.decrypt(encrypted_result, plain_result)
                weight = self.encoder.decode(plain_result)
                weight.resize((partition_weight["partition_size"]))

                weight = numpy.true_divide(numpy.array(weight), num_party).tolist()
                layer_weight.extend(weight)

            update_weights.append(numpy.array(layer_weight))
        return update_weights
