#!/usr/bin/env python3
"""Script to create an inception block"""

import tensorflow.keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Function that builds the ResNet-50 architecture
    Returns: the keras model
    """
    X = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal()

    conv1 = K.layers.Conv2D(filters=64, kernel_size=7, padding='same',
                            strides=2, kernel_initializer=init)(X)
    batch1 = K.layers.BatchNormalization()(conv1)
    relu1 = K.layers.Activation('relu')(batch1)
    pool_1 = K.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(relu1)
    # 1st projection block
    pr_c1 = projection_block(pool_1, [64, 64, 256], 1)
    # 1st identity blocks
    id_conv1 = identity_bloc(pr_c1, [64, 64, 256])
    id_conv2 = identity_block(id_conv1, [64, 64, 256])
    # 2nd projection block
    pr_conv2 = projection_block(id_conv2, [128, 128, 512])
    # 2nd identity blocks
    id_conv3_1 = identity_block(pr_conv2, [128, 128, 512])
    id_conv3_2 = identity_block(id_conv3_1, [128, 128, 512])
    id_conv3_3 = identity_block(id_conv3_2, [128, 128, 512])
    # 3rd projection block
    pr_conv3 = projection_block(idconv3_3, [256, 256, 1024])
    # 3rd identity blocks
    id_conv4_1 = identity_block(pr_conv3, [256, 256, 1024])
    id_conv4_2 = identity_block(id_conv4_1, [256, 256, 1024])
    id_conv4_3 = identity_block(id_conv4_2, [256, 256, 1024])
    id_conv4_4 = identity_block(id_conv4_3, [256, 256, 1024])
    id_conv4_5 = identity_block(id_conv4_4, [256, 256, 1024])
    # 4th projection block
    pr_conv4 = projection_block(id_conv4_5, [512, 512, 2048])
    # 4th identity blocks
    id_conv5_1 = identity_block(pr_conv4, [512, 512, 2048])
    id_conv5_2 = identity_block(id_conv5_1, [512, 512, 2048])
    # average pool
    avg_pool = K.layers.AveragePooling2D(pool_size=7,
                                         padding='same')(id_conv5_2)
    softmax_ = K.layers.Dense(1000, activation='softmax',
                              kernel_initializer=init)(avg_pool)
    model = K.models.Model(inputs=X, outputs=softmax_)

    return model
