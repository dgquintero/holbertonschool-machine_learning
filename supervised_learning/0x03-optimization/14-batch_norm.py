#!/usr/bin/env python3
"""create_batch_norm_layer function"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    creates a batch normalization layer for a NN
    Arguments:
        prev: is the activated output of the previous layer
        n: is the number of nodes in the layer to be created
        activation: is the activation function
    Returns: tensor of the activated output for the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    x = tf.layers.Dense(units=n, activation=None, kernel_initializer=init)
    x_prev = x(prev)
    scale = tf.Variable(tf.constant(1.0, shape=[n]), name='gamma')
    mean, variance = tf.nn.moments(x_prev, axes=[0])
    beta = tf.Variable(tf.constant(0.0, shape=[n]), name='beta')
    epsilon = 1e-8

    normalization = tf.nn.batch_normalization(x_prev,
                                              mean,
                                              variance,
                                              beta,
                                              scale,
                                              epsilon)
    return activation(normalization)
