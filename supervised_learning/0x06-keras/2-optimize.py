#!/usr/bin/env python3
"""function optimize_model"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Function that sets up Adam optimization 
        Arguments:
            network: the model to optimize
            alpha: the learning rate
            beta1: the first Adam optimization parameter
            beta2: the second Adam optimization parameter
        Returns: None
    """
    A = K.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(optimizer=A, loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return None
