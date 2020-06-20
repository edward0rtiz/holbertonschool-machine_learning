#!/usr/bin/env python3
"""Script to create an inception block"""

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """

    Args:
        A_prev: the output from the previous layer
        filters: tuple or list containing the following filters
                 F11: the number of filters in the 1st 1x1 convolution
                 F3: the number of filters in the 3x3 convolution
                 F12: the number of filters in the 2nd 1x1 convolution
        s: stride of the first convolution in both the main path and
           the shortcut connection
    Returns: the activated output of the projection block
    """
    init = K.initializers.he_normal()
    activation = 'relu'
    F11, F3, F12 = filters

    # First component of the main path
    conv1 = K.layers.Conv2D(filters=F11, kernel_size=1, strides=s,
                            padding='same',
                            kernel_initializer=init)(A_prev)

    batchc1 = K.layers.BatchNormalization(axis=3)(conv1)

    relu1 = K.layers.Activation('relu')(batchc1)

    # Second component of the main path
    conv2 = K.layers.Conv2D(filters=F3, kernel_size=3, padding='same',
                            kernel_initializer=init)(relu1)

    batchc2 = K.layers.BatchNormalization(axis=3)(conv2)

    relu2 = K.layers.Activation('relu')(batchc2)

    # Third component of the main path
    conv3 = K.layers.Conv2D(filters=F12, kernel_size=1, padding='same',
                            kernel_initializer=init)(relu2)

    conv1_proj = K.layers.Conv2D(filters=F12, kernel_size=1, strides=s,
                                 padding='same',
                                 kernel_initializer=init)(A_prev)

    batch3 = K.layers.BatchNormalization(axis=3)(conv3)

    batch4 = K.layers.BatchNormalization(axis=3)(conv1_proj)
    # Add shortcut to main path, pass through a relu activation
    add = K.layers.Add()([batch3, batch4])

    final_relu = K.layers.Activation('relu')(add)

    return final_relu
