#!/usr/bin/env python3
"""create_RMSProp_op function"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    that training operation for a neural network in tensorflow
    using the RMSProp optimization algorithm
    Arguments:
        loss: is the loss of the network
        alpha: is the learning rate
        beta2: isnthe RMSProp weight
        epsilon: is a small number to avoid division by zero
    Returns: RMSProp optimization operation
    """
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                          decay=beta2,
                                          epsilon=epsilon).minimize(loss)
    return optimizer
