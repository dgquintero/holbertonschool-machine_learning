#!/usr/bin/env python3
"""function build_model"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    function that builds a neural network with the Keras library
        Arguments:
            nx: is the number of input features to the network
            layers: list containing the number of nodes in
                each layer of the network
            activations: list containing the activation functions
                used for each layer
            lambtha: is the L2 regularization parameter
            keep_prob: is the probability that a node will be kept for dropout
        Returns: the keras model
    """
    inputs = K.Input(shape=(nx,))
    L2 = K.regularizers.l2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            output = K.layers.Dense(layers[i],
                                    activation=activations[i],
                                    kernel_regularizer=L2)(inputs)
        else:
            dropout = K.layers.Dropout(1 - keep_prob)(output)
            output = K.layers.Dense(layers[i],
                                    activation=activations[i],
                                    kernel_regularizer=L2)(dropout)
    return K.models.Model(inputs=inputs, outputs=output)
