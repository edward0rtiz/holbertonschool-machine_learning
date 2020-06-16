#!/usr/bin/env python3
"""Script to create an inception block"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Function to create a dense block
    Args:
        X: the output from the previous layer
        nb_filters: integer representing the number of filters in X
        growth_rate: growth rate for the dense block
        layers: number of layers in the dense block

    Returns: The concatenated output of each layer within the Dense
             Block and the number of filters within the concatenated
             outputs, respectively
    """
    init = K.initializers.he_normal()

    for i in range(layers):

        batch1 = K.layers.BatchNormalization()(X)

        relu1 = K.layers.Activation('relu')(batch1)

        bottleneck = K.layers.Conv2D(filters=4*growth_rate,
                                     kernel_size=1, padding='same',
                                     kernel_initializer=init)(relu1)

        batch2 = K.layers.BatchNormalization()(bottleneck)
        relu2 = K.layers.Activation('relu')(batch2)

        X_conv = K.layers.Conv2D(filters=growth_rate, kernel_size=3,
                                 padding='same',
                                 kernel_initializer=init)(relu2)
        X = K.layers.concatenate([X, X_conv])
        nb_filters += growth_rate

    return X, nb_filters
