#!/usr/bin/env python3
"""Script for forward prop in tensorflow"""


create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Forward propagation method using TF
    Args:
        x: Input data (placeholder)
        layer_sizes: type list are the n nodes inside the layers
        activations: type list with the activation function per layer

    Returns: Prediction of a DNN

    """
    layer = create_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        layer = create_layer(layer, layer_sizes[i], activations[i])
    return layer
