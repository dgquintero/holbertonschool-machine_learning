#!/usr/bin/env python3
"""Create_layer function"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Arguments:
        prev: tenser output of the previous layer
        n: the number of nodes in the layer to create
        activation: activation that the layer should use
    Returns: the tensor output of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init,
                            name='layer')
    return layer(prev)
