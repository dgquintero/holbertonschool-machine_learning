#!/usr/bin/env python3
"""function save_model and load_model"""
import tensorflow.keras as K


def save_model(network, filename):
    """ saves an entire model
        Arguments:
            network: the model to save
            filename: the path of the file that the model
                shuld be saved
        Returns: None and the loaded model
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    loads an entire mode
    Arguments:
        filename: the path of the file that the model
                shuld be loaded
        Returns: the loaded model
    """
    return K.models.load_model(filename)
