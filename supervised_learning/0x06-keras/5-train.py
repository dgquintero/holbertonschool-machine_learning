#!/usr/bin/env python3
"""function optimize_model"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Function that trains a model using mini-batch gradient descent
        Arguments:
            network: model to train
            data: contain the input data
            labels: containing the labels of data
            batch_size: size of the batch used for mini-batch gd
            epochs: number of passes through data for mini-batch gd
            verbose: boolean if output should be printed during training
            shuffle: boolean that determines whether to shuffle the batches
                every epoch
        Returns: History object
    """
    if validation_data:
        validation_data = validation_data
    else:
        validation_data = None

    history = network.fit(x=data, y=labels, epochs=epochs,
                          validation_data=validation_data,
                          batch_size=batch_size, verbose=verbose,
                          shuffle=shuffle)
    return history
