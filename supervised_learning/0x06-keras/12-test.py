#!/usr/bin/env python3
"""Script to test a DNN using keras"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Function to test a model using keras
    Args:
        network: the network model to test
        data: the input data to test the model with
        labels: are the correct one-hot labels of data
        verbose: boolean that determines if output should be printed during
                 the testing process
    Returns: The loss and accuracy of the model with the testing data.
    """
    evaluation = network.evaluate(data, labels, verbose=verbose)
    return evaluation
