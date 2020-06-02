#!/usr/bin/env python3
"""Script for build a DNN in keras"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Function to create a DNN using keras
    Args:
        nx: number of input features to the network
        layers: list containing the number of nodes in each layer of the network
        activations: list containing the activation functions used for each layer of the network
        lambtha: L2 regularization parameter
        keep_prob: probability that a node will be kept for dropout
    Returns: Keras model

    """

    inputs = K.Input(shape=(nx,))
    for i in range(len(layers)):
        if i == 0:
            output = K.layers.Dense(layers[i],
                                    activation=activations[i],
                                    kernel_regularizer=K.regularizers.l2(lambtha))(inputs)
        else:
            dropout = K.layers.Dropout(keep_prob)(output)
            output = K.layers.Dense(layers[i], activation=activations[i],
                                    kernel_regularizer=K.regularizers.l2(lambtha))(dropout)
    return K.models.Model(inputs=inputs, outputs=output)
