#!/usr/bin/env python3
"""Script to create an inception block"""

import tensorflow.keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """

    Args:
        growth_rate:
        compression:

    Returns:

    """
    init = K.initializers.he_normal()

    X = K.Input(shape=(224, 224, 3))

    bacth0 = K.layers.BatchNormalization()(X)
    relu0 = K.layers.Activation('relu')(bacth0)

    conv1 = K.layers.Conv2D(filters=2*growth_rate, kernel_size=7,
                            strides=2, padding='same',
                            kernel_initializer=init)(relu0)

    max_pool1 = K.layers.MaxPooling2D(pool_size=3, strides=2,
                                      padding='same')(conv1)

    dense1, nb_f1 = dense_block(max_pool1, 2*growth_rate, growth_rate, 6)
    trans1, nb_f2 = transition_layer(dense1, nb_f1, compression)

    dense2, nb_f3 = dense_block(trans1, nb_f2, growth_rate, 12)
    trans2, nb_f4 = transition_layer(dense2, nb_f3, compression)

    dense3, nb_f5 = dense_block(trans2, nb_f4, growth_rate, 24)
    trans3, nb_f6 = transition_layer(dense3, nb_f5, compression)

    dense4, nb_f7 = dense_block(trans3, nb_f6, growth_rate, 16)

    avg_pool = K.layers.AveragePooling2D(pool_size=7, padding='same')(dense4)

    FC = K.layers.Dense(1000, activation='softmax',
                        kernel_initializer=init)(avg_pool)

    model = K.models.Model(inputs=X, outputs=FC)
    return model
