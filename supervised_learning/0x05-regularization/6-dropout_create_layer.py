#!/usr/bin/env python3
""" dropout_create_layer function"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    that creates a layer of a neural network using dropout
    Arguments:
        prev: is a tensor containing the output of the previous layer
        n: the number of nodes the new layer should contain
        activation: the activation function that should be used on the layer
        keep_prob: the probability that a node will be kept
    Returns: the output of the new layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    regularizer = tf.layers.Dropout(keep_prob)
    tensor = tf.layers.Dense(n,
                             activation,
                             kernel_initializer=init,
                             kernel_regularizer=regularizer,
                             name='layer')
    return tensor(prev)

