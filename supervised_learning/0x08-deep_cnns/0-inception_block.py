"""inception_block function"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Function that builds an inception block
    Arguments:
        A_prev: The output from the previous layer
        filters: Tuple or list containing the following filters
                 F1: is the number of filters in the 1x1 convolution
                 F3R: is the number of filters in the 1x1 convolution
                      before the 3x3 convolution
                 F3: is the number of filters in the 3x3 convolution
                 F5R: is the number of filters in the 1x1 convolution
                      before the 5x5 convolution
                 F5: is the number of filters in the 5x5 convolution
                 FPP: is the number of filters in the 1x1 convolution
                      after the max pooling
    Returns: the concatenated output of the inception block
    """
    init = K.initializers.he_normal(seed=None)
    F1, F3R, F3, F5R, F5, FPP = filters
    conv1x1 = K.layers.Conv2D(filters=F1, kernel_size=1, padding='same',
                              activation='relu',
                              kernel_initializer=init)(A_prev)

    conv3x3 = K.layers.Conv2D(filters=F3R, kernel_size=1, padding='same',
                              activation='relu',
                              kernel_initializer=init)(A_prev)

    conv3x3 = K.layers.Conv2D(filters=F3, kernel_size=3, padding='same',
                              activation='relu',
                              kernel_initializer=init)(conv3x3)

    conv5x5 = K.layers.Conv2D(filters=F5R, kernel_size=1, padding='same',
                              activation='relu',
                              kernel_initializer=init)(A_prev)

    conv5x5 = K.layers.Conv2D(filters=F5, kernel_size=5, padding='same',
                              activation='relu',
                              kernel_initializer=init)(conv5x5)

    poolP = K.layers.MaxPooling2D(pool_size=[3, 3], strides=(1, 1),
                                  padding='same')(A_prev)

    poolP = K.layers.Conv2D(filters=FPP, kernel_size=1, padding='same',
                            activation='relu', kernel_initializer=init)(poolP)

    mid_l = K.layers.concatenate([conv1x1, conv3x3, conv5x5, poolP], axis = 3)

    return mid_l
