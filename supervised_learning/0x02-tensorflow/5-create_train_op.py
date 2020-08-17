#!/usr/bin/env python3
"""create_train_op function"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """
    creates the training operation for the network
    Arguments:
        loss:  loss of the network’s prediction
        alpha: learning rate
    Returns: an operation that trains the network using gradient descent
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(loss)
    return train
