#!/usr/bin/env python3
"""Script to create an inception block"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Function to create an inception block
    Args:
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
    activation = 'relu'
    init = K.initializers.he_normal(seed=None)
    F1, F3R, F3, F5R, F5, FPP = filters

    convly_1 = K.layers.Conv2D(filters=F1, kernel_size=1, padding='same',
                               activation=activation,
                               kernel_initializer=init)(A_prev)

    convly_2B = K.layers.Conv2D(filters=F3R, kernel_size=1, padding='same',
                                activation=activation,
                                kernel_initializer=init)(A_prev)

    convly_2 = K.layers.Conv2D(filters=F3, kernel_size=3, padding='same',
                               activation=activation,
                               kernel_initializer=init)(convly_2B)

    convly_3B = K.layers.Conv2D(filters=F5R, kernel_size=1, padding='same',
                                activation=activation,
                                kernel_initializer=init)(A_prev)

    convly_3 = K.layers.Conv2D(filters=F5, kernel_size=5, padding='same',
                               activation=activation,
                               kernel_initializer=init)(convly_3B)

    layer_pool = K.layers.MaxPooling2D(pool_size=[3, 3], strides=(1, 1),
                                       padding='same')(A_prev)

    layer_poolB = K.layers.Conv2D(filters=FPP, kernel_size=1, padding='same',
                                  activation=activation,
                                  kernel_initializer=init)(layer_pool)

    mid_layer = K.layers.concatenate([convly_1, convly_2,
                                      convly_3, layer_poolB])

    return mid_layer
