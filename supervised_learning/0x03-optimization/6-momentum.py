#!/usr/bin/env python3
"""create_momentum_op function"""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """

    Arguments:
        alpha: is the learning rate
        beta1: is the momentum weight
        loss: is the loss of the network
    Returns: the momentum optimization operation
    """
    optimizer = tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
    return optimizer
