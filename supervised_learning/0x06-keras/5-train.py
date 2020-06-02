#!/usr/bin/env python3
"""Script to train a model using keras"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Function to train a model using keras and validate data
    Args:
        network: model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray of shape (m, classes) containing
                the labels of data
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data for mini-batch gradient descent
        validation_data: data to validate the model with, if not None
        verbose: boolean that determines if output should be printed during
                 training
        shuffle: boolean that determines whether to shuffle the batches every
                 epoch.
    Returns: History object generated after training the model

    """
    if validation_data:
        validation_data = validation_data
    else:
        validation_data = None

    return network.fit(x=data, y=labels, batch_size=batch_size,
                       epochs=epochs,
                       validation_data=validation_data,
                       verbose=verbose,
                       shuffle=shuffle)
