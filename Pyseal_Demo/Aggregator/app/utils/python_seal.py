import base64
import os

import numpy
from seal import *
from seal import Evaluator


class PySeal:
    evaluator: Evaluator
    parms: EncryptionParameters

    def __init__(self, seal_server_params, cipher_save_path):
        self.parms = EncryptionParameters(scheme_type.CKKS)
        self.cipher_save_path = cipher_save_path

        # Set Coeff modulus
        coeff_modulus = [SmallModulus(i) for i in seal_server_params["coeff_modulus"]]
        self.parms.set_coeff_modulus(coeff_modulus)

        # Set Plain modulus
        self.parms.set_plain_modulus(seal_server_params["plain_modulus"])

        # Set Poly modulus degree
        self.parms.set_poly_modulus_degree(seal_server_params["poly_modulus_degree"])

        self.context = SEALContext.Create(self.parms)
        self.evaluator = Evaluator(self.context)

        # Store the encrypted here
        self._encrypts = []

    def get_param_info(self):
        res = {
            "scheme": self.get_scheme_name(self.parms.scheme()),
            "poly_modulus_degree": self.parms.poly_modulus_degree(),
            "coeff_modulus": [i.value() for i in self.parms.coeff_modulus()],
            "coeff_modulus_size": [i.bit_count() for i in self.parms.coeff_modulus()],
            "plain_modulus": self.parms.plain_modulus().value(),
        }
        return res

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

    def decode64_and_get_cipher_object(self, ins: str):
        decoded = base64.b64decode(ins.encode('utf-8'))
        with open(self.cipher_save_path, 'wb') as f:
            f.write(decoded)
            cipher_object = Ciphertext()
            cipher_object.load(self.context, self.cipher_save_path)
        os.remove(self.cipher_save_path)
        return cipher_object

    def save_encrypted_weight(self, weight):
        self._encrypts.append(weight)

    def aggregate_encrypted_weights(self):
        if len(self._encrypts) == 0:
            return [], 0
        weight = self._encrypts[0]
        num_party = len(self._encrypts)
        if num_party > 1:
            # Iterate through layers
            for layer_idx in range(len(weight)):
                # Iterate through layer partitions
                for layer_partition_idx in range(len(weight[layer_idx])):
                    agg_layer_partition_weights = []
                    # Iterate through results from workers
                    for worker_result in range(num_party):
                        cipher_object = self.decode64_and_get_cipher_object(
                            self._encrypts[worker_result][layer_idx][layer_partition_idx]["encrypted_content"])
                        agg_layer_partition_weights.append(cipher_object)
                    # print(agg_layer_weights)
                    res = self.add_encrypted_matrix(*agg_layer_partition_weights)
                    weight[layer_idx][layer_partition_idx]["encrypted_content"] = self.save_cipher_and_encode64(res)
                    # print("weight agg:", weight[layer_idx])
        del self._encrypts[:num_party]
        return weight, num_party

    def add_encrypted_matrix(self, *argv):
        inputs = []
        for arg in argv:
            inputs.append(arg)

        if len(inputs) <= 1:  # Case 1: If there is only one input matrix
            return inputs[0]

        # Case 2: If there are at least 2 inputs
        encrypted_result = Ciphertext()
        self.evaluator.add(inputs[0], inputs[1], encrypted_result)
        for i in range(2, len(inputs)):
            self.evaluator.add_inplace(encrypted_result, inputs[i])

        res = encrypted_result
        return res
