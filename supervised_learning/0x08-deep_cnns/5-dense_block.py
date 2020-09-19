#!/usr/bin/env python3
"""dense_block function
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Program that builds a dense block
    """
    init = K.initializers.he_normal()
    for i in range(layers):
        batch1 = K.layers.BatchNormalization()(X)
        act1 = K.layers.Activation(K.activations.relu)(batch1)
        conv = K.layers.Conv2D(growth_rate * 4, (1, 1), padding='same',
                               strides=1, kernel_initializer=init)(act1)
        batch2 = K.layers.BatchNormalization()(conv)
        act2 = K.layers.Activation(K.activations.relu)(batch2)
        X_conv = K.layers.Conv2D(growth_rate, (3, 3), padding='same',
                                 strides=1, kernel_initializer=init)(act2)
        X = K.layers.concatenate([X, X_conv])
        nb_filters += growth_rate
    return X, nb_filters
