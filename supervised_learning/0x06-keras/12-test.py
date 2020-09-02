#!/usr/bin/env python3
"""function test_model"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    function that tests a neural network
    Arguments:
        network: the network model to test
        data: the input data to test the model with
        labels: are the correct one-hot labels of data
        verbose: is a boolean that determines if output
            should be printed during the testing process
    Returns: the loss and accuracy of the model with the
    testing data, respectively
    """
    evaluation = network.evaluate(data, labels, verbose=verbose)
    return evaluation
