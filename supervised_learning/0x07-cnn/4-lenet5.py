#!/usr/bin/env python3
""" lenet5 function"""
import tensorflow as tf


def lenet5(x, y):
    """
    Function that builds a modified version of the LeNet-5 architecture
    Arguments:
        x: (m, 28, 28, 1) containing the input images
        y: (m, 10) containing the one-hot labels
    Returns:
        a tensor for the softmax activated output
        a training operation that utilizes Adam optimization
        a tensor for the loss of the netowrk
        a tensor for the accuracy of the network
    """
    init = tf.contrib.layers.variance_scaling_initializer()
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=6, # Number of filters.
        kernel_size=5, # Size of each filter is 5x5.
        padding="valid", # No padding is applied to the input.
        activation=tf.nn.relu
        kernel_initializer=init)
