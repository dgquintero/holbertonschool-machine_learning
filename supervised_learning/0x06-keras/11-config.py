#!/usr/bin/env python3
"""function save_weights and load_weights"""
import tensorflow.keras as K


def save_config(network, filename):
    """ saves a model’s configuration in JSON format
        Arguments:
            network: is the model whose weights should be saved
            filename: is the path of the file that the weight
                should be saved to
            save_format: is the format in which the weights
                should be saved
        Returns: None
    """
    with open(filename, 'w') as f:
        f.write(network.to_json())
    return None


def load_config(filename):
    """
    loads a model with a specific configuration
    Arguments:
        filename: the path of the file that the model
            shuld be loaded
        Returns: the loaded model
    """
    with open(filename, 'r') as f:
        net_config = f.read()
    return K.models.model_from_json(net_config)
