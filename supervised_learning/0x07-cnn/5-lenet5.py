#!/usr/bin/env python3
""" lenet5 function"""
import tensorflow.keras as K


def lenet5(X):
    """
    Function that builds a modified version of the LeNet-5 architecture
    Arguments:
        x: (m, 28, 28, 1) containing the input images
            m: number of images
    Returns: K.model compiled to use Adam optimization and
        accuracy metrics
    """
    # Convolutional layer 1
    init = K.initializers.he_normal()
    conv1 = K.layers.Conv2D(
        filters=6,   # Number of filters.
        kernel_size=5,    # Size of each filter is 5x5.
        padding="same",     # No padding is applied to the input.
        activation='relu',
        kernel_initializer=init)(X)

    # Pooling Layer #1
    # Sampling half the output of previous layer
    # Output: 14 * 14 * 6
    pool1 = K.layers.MaxPooling2D(
        pool_size=[2, 2],
        strides=2)(conv1)
    # Convolutional Layer #2
    # Output: 10 * 10 * 16
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=5,
        padding="valid",
        activation='relu',
        kernel_initializer=init)(pool1)
    # Pooling Layer #2
    # Output: 5 * 5 * 16
    pool2 = K.layers.MaxPooling2D(
        pool_size=[2, 2],
        strides=2)(conv2)

    # Reshaping output into a single dimention array for input
    # to fully connected layer
    pool2_flat = K.layers.Flatten()(pool2)

    # Fully connected layer #1: Has 120 neurons
    dense1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=init)(pool2_flat)
    # Fully connected layer #2: Has 84 neurons
    dense2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=init)(dense1)
    # Output layer, 10 neurons for each digit
    logits = K.layers.Dense(units=10,
                            kernel_initializer=init,
                            activation='softmax')(dense2)
    # model
    model = K.models.Model(X, logits)

    # Use adam optimizer to reduce cost
    adam = K.optimizers.Adam()

    # compile model
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
