#!/usr/bin/env python3
"""Script to predict a DNN using keras"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Function that makes a prediction using keras
    Args:
        network: network model to make the prediction with
        data: input data to make the prediction with
        verbose: boolean that determines if output should
                be printed during the prediction process
    Returns: The prediction of the data
    """
    prediction = network.predict(data, verbose=verbose)
    return prediction
