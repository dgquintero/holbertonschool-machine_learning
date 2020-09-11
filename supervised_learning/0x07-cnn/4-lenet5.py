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
    # Convolutional layer 1
    init = tf.contrib.layers.variance_scaling_initializer()
    conv1 = tf.layers.Conv2d(
        inputs=x,
        filters=6,   # Number of filters.
        kernel_size=5,    # Size of each filter is 5x5.
        padding="valid",     # No padding is applied to the input.
        activation=tf.nn.relu
        kernel_initializer=init)

    # Pooling Layer #1
    # Sampling half the output of previous layer
    # Output: 14 * 14 * 6
    pool1 = tf.layers.MaxPooling2D(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2)
    # Convolutional Layer #2
    # Output: 10 * 10 * 16
    conv2 = tf.layers.Conv2D(
        inputs=pool1,
        filters=16,
        kernel_size=5,
        padding="valid",
        activation=tf.nn.relu,
        kernel_initializer=init
    )
    # Pooling Layer #2
    # Output: 5 * 5 * 16
    pool2 = tf.layers.MaxPooling2D(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2
    )

    # Reshaping output into a single dimention array for input
    # to fully connected layer
    pool2_flat = tf.layers.Flatten(inputs=pool2)

    # Fully connected layer #1: Has 120 neurons
    dense1 = tf.layers.Dense(inputs=pool2_flat,
                             units=120,
                             activation=tf.nn.relu
                             kernel_initializer=init)
    # Fully connected layer #2: Has 84 neurons
    dense2 = tf.layers.Dense(inputs=dense1,
                             units=84,
                             activation=tf.nn.relu,
                             kernel_initializer=init)
    # Output layer, 10 neurons for each digit
    logits = tf.layers.Dense(inputs=dense2, units=10, kernel_initializer=init)

    # prediction variable
    y_predict = logits

    # Compute the cross-entropy loss fucntion
    loss = tf.losses.softmax_cross_entropy(y, y_predict)

    # Use adam optimizer to reduce cost
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # For testing and prediction
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    y_pred = tf.nn.softmax(y_pred)

    return y_pred, optimizer, loss, accuracy
