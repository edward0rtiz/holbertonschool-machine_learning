#!/usr/bin/env python3
"""Script to save and load JSON configuration of a keras model"""


import tensorflow.keras as K


def save_config(network, filename):
    """
    Function to save the configuration model in JSON format
    Args:
        network: model whose configuration should be saved
        filename: path of the file that the configuration should be saved to
    Returns: None
    """
    with open(filename, 'w') as f:
        f.write(network.to_json())
    return None


def load_config(filename):
    """
    Function to load the configuration model in JSON format
    Args:
        filename: path of the file containing the model’s configuration
                  in JSON format

    Returns: the loaded model
    """
    with open(filename, 'r') as f:
        network_config = f.read()
    return K.models.model_from_json(network_config)
