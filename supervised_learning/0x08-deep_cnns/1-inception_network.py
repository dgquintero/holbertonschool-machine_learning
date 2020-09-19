#!/usr/bin/env python3
""" inception_network function"""

import tensorflow.keras as K

inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Function hat builds an inception network
    Retunrs: the keras model
    """
    X = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal()

    conv_1 = K.layers.Conv2D(filters=64, kernel_size=7, strides=(2, 2),
                             padding='same', activation='relu',
                             kernel_initializer=init)(X)
    max_pool_1 = K.layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2),
                                       padding='same')(conv_1)
    conv_2P = K.layers.Conv2D(filters=64, kernel_size=1, padding='valid',
                              activation='relu',
                              kernel_initializer=init)(max_pool_1)
    conv_2 = K.layers.Conv2D(filters=192, kernel_size=3, padding='same',
                             activation='relu',
                             kernel_initializer=init)(conv_2P)
    max_pool_2 = K.layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2),
                                       padding='same')(conv_2)

    inception1 = inception_block(max_pool_2, [64, 96, 128, 16, 32, 32])
    inception2 = inception_block(inception1, [128, 128, 192, 32, 96, 64])

    max_pool_3 = K.layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2),
                                       padding='same')(inception2)

    inception3 = inception_block(max_pool_3, [192, 96, 208, 16, 48, 64])
    inception4 = inception_block(inception3, [160, 112, 224, 24, 64, 64])
    inception5 = inception_block(inception4, [128, 128, 256, 24, 64, 64])
    inception6 = inception_block(inception5, [112, 144, 288, 32, 64, 64])
    inception7 = inception_block(inception6, [256, 160, 320, 32, 128, 128])

    max_pool_4 = K.layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2),
                                       padding='same')(inception7)

    iblock_8 = inception_block(max_pool_4, [256, 160, 320, 32, 128, 128])
    iblock_9 = inception_block(iblock_8, [384, 192, 384, 48, 128, 128])

    avgPool = K.layers.AveragePooling2D(pool_size=[7, 7], strides=(1, 1),
                                        padding='valid')(iblock_9)

    dropout = K.layers.Dropout(.4)(avgPool)

    softmax_ = K.layers.Dense(1000, activation='softmax',
                              kernel_initializer=init)(dropout)

    model = K.models.Model(inputs=X, outputs=softmax_)

    return model
