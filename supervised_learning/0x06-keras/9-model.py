#!/usr/bin/env python3
"""Script to save and load a keras model"""


import tensorflow.keras as K


def save_model(network, filename):
    """
    Function to save a model
    Args:
        network: model to save
        filename: the path of the file that the model should be saved to

    Returns: None
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    Function to load a model
    Args:
        filename: The path of the file that the model should be loaded from

    Returns: the loadel model
    """
    return K.models.load_model(filename)
