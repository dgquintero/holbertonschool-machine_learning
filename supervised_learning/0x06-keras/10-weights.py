#!/usr/bin/env python3
"""function save_weights and load_weights"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """ saves a model's weights
        Arguments:
            network: is the model whose weights should be saved
            filename: is the path of the file that the weight
                should be saved to
            save_format: is the format in which the weights
                should be saved
        Returns: None
    """
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """
    loads model's weights
    Arguments:
        network: is the model to which the weights
            should be loaded
        filename: the path of the file that the model
            shuld be loaded
        Returns: None
    """
    network.load_weights(filename)
    return None
